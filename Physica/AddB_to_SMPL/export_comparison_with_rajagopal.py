#!/usr/bin/env python3
"""
Export SMPL mesh + Rajagopal anatomical mesh + skeleton for MeshLab comparison
This provides much better visualization than geometric primitives
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import sys
import trimesh
import pickle
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Import SMPL model
sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel

try:
    import nimblephysics as nimble
except ImportError as exc:
    raise RuntimeError(
        "nimblephysics is required. Install in physpt environment."
    ) from exc


def transform_smpl_poses_to_yup(poses: np.ndarray, trans: np.ndarray):
    """
    Transform SMPL poses and translations from Z-up to Y-up coordinate system

    IMPORTANT: This must be done BEFORE SMPL forward pass!

    Why: SMPL's Linear Blend Skinning (LBS) computes vertex deformations
    based on bone rotations. If we transform the OUTPUT vertices instead
    of INPUT poses, the skinning happens in the wrong coordinate system,
    causing severe mesh distortion.

    Args:
        poses: (N, 24, 3) axis-angle rotations in Z-up
        trans: (N, 3) translations in Z-up

    Returns:
        poses_yup: (N, 24, 3) axis-angle rotations in Y-up
        trans_yup: (N, 3) translations in Y-up
    """
    # Global rotation: Z-up → Y-up (rotate -90° around X-axis)
    global_rot = R.from_euler('x', -90, degrees=True)

    # Handle single frame or batch
    if poses.ndim == 2:  # Single frame (24, 3)
        poses = poses.reshape(1, 24, 3)
        trans = trans.reshape(1, 3)
        single_frame = True
    else:
        single_frame = False

    num_frames = poses.shape[0]
    poses_yup = np.zeros_like(poses)

    # Transform each joint's rotation
    for frame_idx in range(num_frames):
        for joint_idx in range(24):
            # Get current rotation as scipy Rotation object
            joint_rot = R.from_rotvec(poses[frame_idx, joint_idx])
            # Apply global rotation: R_yup = R_global * R_zup
            new_rot = global_rot * joint_rot
            # Store as axis-angle
            poses_yup[frame_idx, joint_idx] = new_rot.as_rotvec()

    # Transform translations
    trans_yup = global_rot.apply(trans.reshape(-1, 3))

    if single_frame:
        poses_yup = poses_yup[0]  # (24, 3)
        trans_yup = trans_yup[0]  # (3,)

    return poses_yup, trans_yup


def load_rajagopal_template(mesh_path: Path, segmentation_path: Path):
    """
    Load Rajagopal template mesh and bone segmentation

    Returns:
        template_mesh: trimesh.Trimesh with ~85k vertices
        bone_segmentation: dict mapping bone_name -> vertex_indices
    """
    print(f"  Loading template mesh: {mesh_path.name}")
    template = trimesh.load(str(mesh_path), process=False)

    print(f"  Loading bone segmentation: {segmentation_path.name}")
    with open(segmentation_path, 'rb') as f:
        bone_seg = pickle.load(f)

    print(f"  Template: {len(template.vertices)} vertices, {len(template.faces)} faces")
    print(f"  Bones: {len(bone_seg)} body parts")

    return template, bone_seg


def pose_rajagopal_mesh(
    template: trimesh.Trimesh,
    bone_seg: dict,
    skeleton: 'nimble.dynamics.Skeleton',
    dof_positions: np.ndarray,
    debug: bool = False
) -> trimesh.Trimesh:
    """
    Transform Rajagopal template mesh to match skeleton pose

    Uses relative transforms to avoid double transformation:
    T_relative = T_target @ inv(T_rest)

    Args:
        template: Unposed Rajagopal mesh
        bone_seg: Bone name -> vertex indices mapping
        skeleton: Nimblephysics skeleton
        dof_positions: DOF values for this frame
        debug: Print debug information

    Returns:
        Posed mesh with same topology but transformed vertices
    """
    # Bone name mapping from segmentation to skeleton
    # Map Rajagopal OSSO bone names to AddBiomechanics skeleton names

    # Check if skeleton has arm bones
    body_node_names = {skeleton.getBodyNode(i).getName() for i in range(skeleton.getNumBodyNodes())}
    has_arm_bones = 'humerus_r' in body_node_names or 'humerus_l' in body_node_names

    # STRATEGY: Only use bones with exact skeleton matches, no mapping
    # If No_Arm skeleton, attach arm bones to torso
    bone_name_map = {}

    if not has_arm_bones:
        # No_Arm skeleton: attach arm bones to torso
        bone_name_map.update({
            'scapula_l': 'torso',
            'scapula_r': 'torso',
            'humerus_l': 'torso',
            'radius_l': 'torso',
            'ulna_l': 'torso',
            'hand_l': 'torso',
            'humerus_r': 'torso',
            'radius_r': 'torso',
            'ulna_r': 'torso',
            'hand_r': 'torso',
        })

    # Set skeleton to desired pose
    skeleton.setPositions(dof_positions)

    # Get world transforms for all body nodes
    transforms = {}
    for i in range(skeleton.getNumBodyNodes()):
        node = skeleton.getBodyNode(i)
        transforms[node.getName()] = node.getWorldTransform().matrix()

    if debug:
        print(f"\nSkeleton has {len(transforms)} body nodes")
        print(f"Bone segmentation has {len(bone_seg)} parts")
        print(f"Sample transforms:")
        for name in list(transforms.keys())[:3]:
            T = transforms[name]
            print(f"  {name}: pos = {T[:3, 3]}")

    # Track which vertices to keep (only transformed ones)
    vertices_to_keep = []
    new_vertices = []
    vertex_index_map = {}  # old_idx -> new_idx

    # Transform vertices for each bone and collect only transformed vertices
    for bone_name, vertex_indices in bone_seg.items():
        # Map bone name if needed
        skel_bone_name = bone_name_map.get(bone_name, bone_name)

        if skel_bone_name in transforms:
            T = transforms[skel_bone_name]  # 4x4 homogeneous transform

            # Convert vertices to homogeneous coordinates
            vertices_homo = np.hstack([
                template.vertices[vertex_indices],
                np.ones((len(vertex_indices), 1))
            ])

            # Apply transformation
            transformed = (T @ vertices_homo.T).T[:, :3]

            # Add to new vertex list
            for old_idx, new_vert in zip(vertex_indices, transformed):
                vertex_index_map[old_idx] = len(new_vertices)
                new_vertices.append(new_vert)
                vertices_to_keep.append(old_idx)

            if debug and bone_name in ['pelvis', 'head', 'femur_r', 'humerus_r']:
                orig_center = template.vertices[vertex_indices].mean(axis=0)
                new_center = transformed.mean(axis=0)
                print(f"  {bone_name:15s} ({len(vertex_indices):5d} verts): {orig_center} -> {new_center}")
        else:
            if debug:
                print(f"  REMOVED: {bone_name} ({len(vertex_indices)} verts) - no skeleton match")

    # Convert to numpy array
    new_vertices = np.array(new_vertices)

    # Rebuild faces - only keep faces where all vertices are in the kept set
    vertices_to_keep_set = set(vertices_to_keep)
    new_faces = []

    for face in template.faces:
        if all(v_idx in vertices_to_keep_set for v_idx in face):
            # Remap vertex indices
            new_face = [vertex_index_map[v_idx] for v_idx in face]
            new_faces.append(new_face)

    new_faces = np.array(new_faces)

    if debug:
        print(f"\nKept {len(new_vertices)}/{len(template.vertices)} vertices ({100*len(new_vertices)/len(template.vertices):.1f}%)")
        print(f"Removed {len(template.vertices) - len(new_vertices)} unmapped vertices")
        print(f"Kept {len(new_faces)}/{len(template.faces)} faces ({100*len(new_faces)/len(template.faces):.1f}%)")
        print(f"Original bounds: {template.vertices.min(axis=0)} to {template.vertices.max(axis=0)}")
        print(f"Posed bounds: {new_vertices.min(axis=0)} to {new_vertices.max(axis=0)}")

    # Create new mesh with only transformed vertices
    return trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False
    )


def load_b3d_skeleton(b3d_path: str, processing_pass: int = 0):
    """Load skeleton from .b3d file"""
    subj = nimble.biomechanics.SubjectOnDisk(b3d_path)
    skel = subj.readSkel(processing_pass)
    return subj, skel


def read_b3d_frames(
    b3d_path: str,
    trial: int = 0,
    processing_pass: int = 0,
    start_frame: int = 0,
    num_frames: int = -1
):
    """Read frames from .b3d file"""
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

    return frames


def create_smpl_skeleton_mesh(joints: np.ndarray, sphere_radius: float = 0.012, bone_radius: float = 0.006) -> trimesh.Trimesh:
    """Create SMPL skeleton visualization with spheres at joints and cylinders for bones

    Args:
        joints: [24, 3] array of SMPL joint positions
        sphere_radius: Radius for joint spheres
        bone_radius: Radius for bone cylinders

    Returns:
        Combined mesh with spheres and bone connections
    """
    meshes = []

    # Create spheres at joint positions
    for joint_pos in joints:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
        sphere.vertices += joint_pos
        sphere.visual.vertex_colors = [0, 255, 0, 255]  # Green for SMPL
        meshes.append(sphere)

    # SMPL joint hierarchy (24 joints)
    # 0:Pelvis, 1:L_Hip, 2:R_Hip, 3:Spine1, 4:L_Knee, 5:R_Knee, 6:Spine2, 7:L_Ankle, 8:R_Ankle,
    # 9:Spine3, 10:L_Foot, 11:R_Foot, 12:Neck, 13:L_Collar, 14:R_Collar, 15:Head,
    # 16:L_Shoulder, 17:R_Shoulder, 18:L_Elbow, 19:R_Elbow, 20:L_Wrist, 21:R_Wrist, 22:L_Hand, 23:R_Hand

    bone_pairs = [
        # Spine chain
        (0, 3),    # Pelvis → Spine1
        (3, 6),    # Spine1 → Spine2
        (6, 9),    # Spine2 → Spine3
        (9, 12),   # Spine3 → Neck
        (12, 15),  # Neck → Head

        # Left leg
        (0, 1),    # Pelvis → L_Hip
        (1, 4),    # L_Hip → L_Knee
        (4, 7),    # L_Knee → L_Ankle
        (7, 10),   # L_Ankle → L_Foot

        # Right leg
        (0, 2),    # Pelvis → R_Hip
        (2, 5),    # R_Hip → R_Knee
        (5, 8),    # R_Knee → R_Ankle
        (8, 11),   # R_Ankle → R_Foot

        # Left arm
        (9, 13),   # Spine3 → L_Collar
        (13, 16),  # L_Collar → L_Shoulder
        (16, 18),  # L_Shoulder → L_Elbow
        (18, 20),  # L_Elbow → L_Wrist
        (20, 22),  # L_Wrist → L_Hand

        # Right arm
        (9, 14),   # Spine3 → R_Collar
        (14, 17),  # R_Collar → R_Shoulder
        (17, 19),  # R_Shoulder → R_Elbow
        (19, 21),  # R_Elbow → R_Wrist
        (21, 23),  # R_Wrist → R_Hand
    ]

    # Create cylinders for bone connections
    for start_idx, end_idx in bone_pairs:
        if start_idx < len(joints) and end_idx < len(joints):
            start_pos = joints[start_idx]
            end_pos = joints[end_idx]

            # Calculate bone vector and length
            bone_vec = end_pos - start_pos
            bone_length = np.linalg.norm(bone_vec)

            if bone_length > 1e-6:  # Avoid zero-length bones
                # Create cylinder along z-axis
                cylinder = trimesh.creation.cylinder(
                    radius=bone_radius,
                    height=bone_length,
                    sections=8
                )

                # Rotate cylinder to align with bone direction
                z_axis = np.array([0, 0, 1])
                bone_dir = bone_vec / bone_length

                # Rotation axis and angle
                rotation_axis = np.cross(z_axis, bone_dir)
                rotation_axis_len = np.linalg.norm(rotation_axis)

                if rotation_axis_len > 1e-6:
                    rotation_axis = rotation_axis / rotation_axis_len
                    rotation_angle = np.arccos(np.clip(np.dot(z_axis, bone_dir), -1.0, 1.0))

                    # Create rotation matrix
                    from scipy.spatial.transform import Rotation
                    rotation = Rotation.from_rotvec(rotation_angle * rotation_axis)
                    cylinder.vertices = rotation.apply(cylinder.vertices)
                elif np.dot(z_axis, bone_dir) < 0:
                    # Flip cylinder if pointing opposite direction
                    cylinder.vertices[:, 2] *= -1

                # Translate to midpoint
                midpoint = (start_pos + end_pos) / 2
                cylinder.vertices += midpoint

                # Color bones green (same as SMPL)
                cylinder.visual.vertex_colors = [50, 200, 50, 255]  # Green
                meshes.append(cylinder)

    if meshes:
        return trimesh.util.concatenate(meshes)
    return trimesh.Trimesh()


def create_skeleton_mesh(joint_centers: np.ndarray, sphere_radius: float = 0.015, bone_radius: float = 0.008) -> trimesh.Trimesh:
    """Create skeleton visualization with spheres at joints and cylinders for bones

    Args:
        joint_centers: [N_joints, 3] array of joint positions (20 joints for With_Arm)
        sphere_radius: Radius for joint spheres
        bone_radius: Radius for bone cylinders

    Returns:
        Combined mesh with spheres and bone connections
    """
    meshes = []

    # Create spheres at joint positions
    for joint_pos in joint_centers:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
        sphere.vertices += joint_pos
        sphere.visual.vertex_colors = [255, 0, 0, 255]  # Red
        meshes.append(sphere)

    # Define bone connections for With_Arm skeleton (20 joints)
    # Joint order: pelvis(0), femur_r(1), tibia_r(2), talus_r(3), calcn_r(4), toes_r(5),
    #              femur_l(6), tibia_l(7), talus_l(8), calcn_l(9), toes_l(10),
    #              torso(11), humerus_r(12), ulna_r(13), radius_r(14), hand_r(15),
    #              humerus_l(16), ulna_l(17), radius_l(18), hand_l(19)

    bone_pairs = [
        # Spine (pelvis to torso)
        (0, 11),   # pelvis → torso (척추)

        # Right leg
        (0, 1),    # pelvis → femur_r
        (1, 2),    # femur_r → tibia_r
        (2, 3),    # tibia_r → talus_r
        (3, 4),    # talus_r → calcn_r
        (4, 5),    # calcn_r → toes_r

        # Left leg
        (0, 6),    # pelvis → femur_l
        (6, 7),    # femur_l → tibia_l
        (7, 8),    # tibia_l → talus_l
        (8, 9),    # talus_l → calcn_l
        (9, 10),   # calcn_l → toes_l

        # Right arm
        (11, 12),  # torso → humerus_r (어깨)
        (12, 13),  # humerus_r → ulna_r
        (13, 14),  # ulna_r → radius_r
        (14, 15),  # radius_r → hand_r

        # Left arm
        (11, 16),  # torso → humerus_l (어깨)
        (16, 17),  # humerus_l → ulna_l
        (17, 18),  # ulna_l → radius_l
        (18, 19),  # radius_l → hand_l
    ]

    # Create cylinders for bone connections
    for start_idx, end_idx in bone_pairs:
        if start_idx < len(joint_centers) and end_idx < len(joint_centers):
            start_pos = joint_centers[start_idx]
            end_pos = joint_centers[end_idx]

            # Calculate bone vector and length
            bone_vec = end_pos - start_pos
            bone_length = np.linalg.norm(bone_vec)

            if bone_length > 1e-6:  # Avoid zero-length bones
                # Create cylinder along z-axis
                cylinder = trimesh.creation.cylinder(
                    radius=bone_radius,
                    height=bone_length,
                    sections=8
                )

                # Rotate cylinder to align with bone direction
                # Default cylinder is along z-axis [0, 0, 1]
                z_axis = np.array([0, 0, 1])
                bone_dir = bone_vec / bone_length

                # Rotation axis and angle
                rotation_axis = np.cross(z_axis, bone_dir)
                rotation_axis_len = np.linalg.norm(rotation_axis)

                if rotation_axis_len > 1e-6:
                    rotation_axis = rotation_axis / rotation_axis_len
                    rotation_angle = np.arccos(np.clip(np.dot(z_axis, bone_dir), -1.0, 1.0))

                    # Create rotation matrix
                    from scipy.spatial.transform import Rotation
                    rotation = Rotation.from_rotvec(rotation_angle * rotation_axis)
                    cylinder.vertices = rotation.apply(cylinder.vertices)
                elif np.dot(z_axis, bone_dir) < 0:
                    # Flip cylinder if pointing opposite direction
                    cylinder.vertices[:, 2] *= -1

                # Translate to midpoint
                midpoint = (start_pos + end_pos) / 2
                cylinder.vertices += midpoint

                # Color bones blue
                cylinder.visual.vertex_colors = [100, 150, 255, 255]  # Light blue
                meshes.append(cylinder)

    if meshes:
        return trimesh.util.concatenate(meshes)
    return trimesh.Trimesh()


def create_joint_spheres_mesh(joint_centers: np.ndarray, radius: float = 0.015) -> trimesh.Trimesh:
    """Create red spheres at joint positions (legacy function, use create_skeleton_mesh instead)

    Args:
        joint_centers: [N_joints, 3] array of joint positions
        radius: Sphere radius

    Returns:
        Combined mesh of all joint spheres
    """
    meshes = []
    for joint_pos in joint_centers:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere.vertices += joint_pos
        sphere.visual.vertex_colors = [255, 0, 0, 255]  # Red
        meshes.append(sphere)

    if meshes:
        return trimesh.util.concatenate(meshes)
    return trimesh.Trimesh()


def export_comparison_with_rajagopal(
    b3d_path: Path,
    smpl_params_path: Path,
    smpl_model_path: Path,
    rajagopal_mesh_path: Path,
    rajagopal_seg_path: Path,
    output_dir: Path,
    frame_range: tuple = None,
    smpl_color: tuple = (0.7, 0.8, 1.0, 0.7),
    rajagopal_color: tuple = (1.0, 0.9, 0.8, 0.9),
    export_skeleton: bool = True,
    use_neutral_shape: bool = False
):
    """
    Export SMPL mesh + Rajagopal anatomical mesh for comparison

    Args:
        b3d_path: Path to .b3d file
        smpl_params_path: Path to smpl_params.npz
        smpl_model_path: Path to SMPL model
        rajagopal_mesh_path: Path to Rajagopal template OBJ
        rajagopal_seg_path: Path to bone segmentation PKL
        output_dir: Output directory
        frame_range: Optional (start, end)
        smpl_color: RGBA for SMPL mesh
        rajagopal_color: RGBA for Rajagopal mesh
        export_skeleton: Also export skeleton markers
        use_neutral_shape: If True, use beta=0 (SMPL_neutral) instead of optimized betas
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SMPL + RAJAGOPAL ANATOMICAL MESH EXPORT")
    print("="*80)

    # -------------------------------------------------------------------------
    # STEP 1: Load SMPL Model
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading SMPL model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl_model = SMPLModel(model_path=str(smpl_model_path), device=device)

    if smpl_model.faces is None:
        raise ValueError("SMPL model does not contain face information")

    data = np.load(smpl_params_path)
    betas_np = data['betas']
    poses_np = data['poses']  # (N, 24, 3) in Z-up
    trans_np = data['trans']  # (N, 3) in Z-up

    # Option: Use neutral shape (beta=0) or optimized shape
    if use_neutral_shape:
        print("  Using SMPL_neutral (beta=0) for shape")
        betas_np = np.zeros_like(betas_np)
    else:
        print(f"  Using optimized shape (betas: {betas_np[:5]}...)")

    # CRITICAL: Transform poses and trans to Y-up BEFORE SMPL forward!
    # This ensures skinning happens in the correct coordinate system
    print("  Transforming SMPL poses from Z-up to Y-up coordinate system...")
    poses_yup_np, trans_yup_np = transform_smpl_poses_to_yup(poses_np, trans_np)

    # Convert to tensors
    betas = torch.tensor(betas_np, dtype=torch.float32, device=device)
    poses = torch.tensor(poses_yup_np, dtype=torch.float32, device=device)
    trans = torch.tensor(trans_yup_np, dtype=torch.float32, device=device)

    smpl_num_frames = poses.shape[0]
    smpl_faces = smpl_model.faces.cpu().numpy()

    print(f"  Loaded SMPL params: {smpl_num_frames} frames")

    print(f"  SMPL frames: {smpl_num_frames}")

    # -------------------------------------------------------------------------
    # STEP 2: Load Rajagopal Template
    # -------------------------------------------------------------------------
    print("\n[2/6] Loading Rajagopal template...")
    rajagopal_template, bone_segmentation = load_rajagopal_template(
        rajagopal_mesh_path,
        rajagopal_seg_path
    )

    # -------------------------------------------------------------------------
    # STEP 3: Load AddBiomechanics Skeleton
    # -------------------------------------------------------------------------
    print("\n[3/6] Loading AddBiomechanics skeleton...")
    subj, skeleton = load_b3d_skeleton(str(b3d_path))

    print(f"  Skeleton: {skeleton.getNumBodyNodes()} body nodes")
    print(f"  DOFs: {skeleton.getNumDofs()}")

    # -------------------------------------------------------------------------
    # STEP 4: Read Frames
    # -------------------------------------------------------------------------
    print("\n[4/6] Reading frames from .b3d...")

    start_frame = frame_range[0] if frame_range else 0
    num_frames = (frame_range[1] - start_frame) if frame_range else -1

    frames = read_b3d_frames(
        str(b3d_path),
        trial=0,
        processing_pass=0,
        start_frame=start_frame,
        num_frames=num_frames
    )

    addb_num_frames = len(frames)
    print(f"  AddB frames: {addb_num_frames}")

    # -------------------------------------------------------------------------
    # STEP 5: Determine Export Range
    # -------------------------------------------------------------------------
    print("\n[5/6] Determining export range...")
    max_frames = min(smpl_num_frames, addb_num_frames)
    export_end = max_frames

    print(f"  Exporting {export_end} frames (0 to {export_end-1})")

    # -------------------------------------------------------------------------
    # STEP 6: Export All Meshes
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Exporting meshes...")

    for frame_idx in tqdm(range(export_end), desc="Exporting frames"):
        # Export SMPL mesh and skeleton
        pose = poses[frame_idx].reshape(24, 3)
        t = trans[frame_idx]

        # Debug: Print pose statistics for first frame
        if frame_idx == 0:
            print(f"\n[DEBUG] Frame 0 SMPL inputs (already in Y-up):")
            print(f"  betas shape: {betas.shape}")
            print(f"  pose shape: {pose.shape}")
            print(f"  trans shape: {t.shape}")
            print(f"  trans value: {t}")
            print(f"  pose min/max/mean: {pose.min():.4f} / {pose.max():.4f} / {pose.mean():.4f}")

        # SMPL forward with Y-up poses
        # Output is naturally in Y-up coordinate system!
        with torch.no_grad():
            vertices, joints = smpl_model.forward(betas, pose, t)

        vertices_np = vertices.cpu().numpy()
        joints_np = joints.cpu().numpy()  # [24, 3] SMPL joint positions in Y-up

        if frame_idx == 0:
            print(f"\n[DEBUG] Frame 0 SMPL outputs (Y-up):")
            print(f"  vertices shape: {vertices_np.shape}")
            print(f"  joints shape: {joints_np.shape}")
            print(f"  vertices Y range: [{vertices_np[:, 1].min():.4f}, {vertices_np[:, 1].max():.4f}]")
            print(f"  joints Y range: [{joints_np[:, 1].min():.4f}, {joints_np[:, 1].max():.4f}]")
            print(f"  First 3 joint positions:")
            for i in range(3):
                print(f"    Joint {i}: {joints_np[i]}")

        # Get AddB frame for pelvis alignment
        frame = frames[frame_idx]
        dof_positions = np.array(frame.processingPasses[0].pos)
        joint_centers = np.array(frame.processingPasses[0].jointCenters).reshape(-1, 3)

        # NO coordinate transformation needed!
        # Both SMPL (now Y-up) and AddB (Y-up) are in the same coordinate system
        # Just align pelvis positions
        if frame_idx == 0:
            print(f"\n[DEBUG] Pelvis alignment:")
            print(f"  SMPL pelvis (Y-up): {joints_np[0]}")
            print(f"  AddB pelvis (Y-up): {joint_centers[0]}")

        # Align SMPL skeleton to AddB pelvis position
        translation_offset = joint_centers[0] - joints_np[0]

        if frame_idx == 0:
            print(f"  Translation offset: {translation_offset}")

        # Apply offset to SMPL joints and vertices
        joints_aligned = joints_np + translation_offset
        vertices_aligned = vertices_np + translation_offset

        if frame_idx == 0:
            print(f"  SMPL pelvis (aligned): {joints_aligned[0]}")
            print(f"  SMPL joints Y range (aligned): [{joints_aligned[:, 1].min():.4f}, {joints_aligned[:, 1].max():.4f}]")
            print(f"  SMPL vertices Y range (aligned): [{vertices_aligned[:, 1].min():.4f}, {vertices_aligned[:, 1].max():.4f}]")
            print(f"  AddB joints Y range: [{joint_centers[:, 1].min():.4f}, {joint_centers[:, 1].max():.4f}]")

        # Export SMPL mesh with aligned coordinates
        smpl_mesh = trimesh.Trimesh(
            vertices=vertices_aligned,
            faces=smpl_faces,
            process=False
        )
        smpl_mesh.visual.vertex_colors = [int(c*255) for c in smpl_color]

        actual_frame = start_frame + frame_idx
        smpl_output = output_dir / f"smpl_frame_{actual_frame:04d}.obj"
        smpl_mesh.export(str(smpl_output))

        # Export SMPL skeleton (pose visualization) with aligned coordinates
        smpl_skeleton = create_smpl_skeleton_mesh(joints_aligned, sphere_radius=0.012, bone_radius=0.006)
        smpl_skel_output = output_dir / f"smpl_skeleton_frame_{actual_frame:04d}.obj"
        smpl_skeleton.export(str(smpl_skel_output))

        # Export Rajagopal mesh
        rajagopal_posed = pose_rajagopal_mesh(
            rajagopal_template,
            bone_segmentation,
            skeleton,
            dof_positions,
            debug=(frame_idx == 0)  # Debug first frame only
        )
        rajagopal_posed.visual.vertex_colors = [int(c*255) for c in rajagopal_color]

        rajagopal_output = output_dir / f"rajagopal_frame_{actual_frame:04d}.obj"
        rajagopal_posed.export(str(rajagopal_output))

        # Export AddB skeleton with joints and bones
        if export_skeleton:
            addb_skeleton_mesh = create_skeleton_mesh(joint_centers, sphere_radius=0.015, bone_radius=0.008)
            joints_output = output_dir / f"joints_frame_{actual_frame:04d}.obj"
            addb_skeleton_mesh.export(str(joints_output))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("EXPORT COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total frames exported: {export_end}")
    print(f"\nFiles created:")
    print(f"  SMPL meshes: smpl_frame_XXXX.obj ({export_end} files)")
    print(f"  SMPL skeletons: smpl_skeleton_frame_XXXX.obj ({export_end} files)")
    print(f"  Rajagopal meshes: rajagopal_frame_XXXX.obj ({export_end} files)")
    if export_skeleton:
        print(f"  AddB skeletons: joints_frame_XXXX.obj ({export_end} files)")
    print(f"\n" + "="*80)
    print("HOW TO VISUALIZE IN MESHLAB:")
    print("="*80)
    print(f"\n1. Open MeshLab")
    print(f"\n2. Load all models for a single frame:")
    print(f"   File -> Import Mesh -> Select:")
    print(f"     - {output_dir}/smpl_frame_0000.obj (SMPL mesh - blue)")
    print(f"     - {output_dir}/smpl_skeleton_frame_0000.obj (SMPL skeleton - green)")
    print(f"     - {output_dir}/rajagopal_frame_0000.obj (AddB mesh - skin color)")
    if export_skeleton:
        print(f"     - {output_dir}/joints_frame_0000.obj (AddB skeleton - red/blue)")
    print(f"\n3. Compare SMPL vs AddB alignment!")
    print(f"\n4. Tips:")
    print(f"   - Toggle visibility with Layer Dialog (Alt+L)")
    print(f"   - Adjust transparency in Render menu")
    print(f"   - Green skeleton = SMPL pose")
    print(f"   - Red/blue skeleton = AddBiomechanics ground truth")
    print()

    return export_end


def main():
    parser = argparse.ArgumentParser(
        description="Export SMPL + Rajagopal anatomical mesh for MeshLab comparison"
    )
    parser.add_argument(
        '--b3d',
        type=str,
        required=True,
        help='Path to .b3d file'
    )
    parser.add_argument(
        '--smpl_params',
        type=str,
        required=True,
        help='Path to smpl_params.npz'
    )
    parser.add_argument(
        '--smpl_model',
        type=str,
        default='models/smpl_model.pkl',
        help='Path to SMPL model'
    )
    parser.add_argument(
        '--rajagopal_mesh',
        type=str,
        default='/egr/research-zijunlab/kwonjoon/Code/SMPL2AddBiomechanics-main/SMPL2AddBiomechanics-main/smpl2ab/data/osso_rajagopal_unposed_v2.obj',
        help='Path to Rajagopal template mesh'
    )
    parser.add_argument(
        '--rajagopal_seg',
        type=str,
        default='/egr/research-zijunlab/kwonjoon/Code/SMPL2AddBiomechanics-main/SMPL2AddBiomechanics-main/smpl2ab/data/OSSO_osim_bone_groups_to_vertices_v4.pkl',
        help='Path to bone segmentation'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--start_frame',
        type=int,
        default=None,
        help='Start frame'
    )
    parser.add_argument(
        '--end_frame',
        type=int,
        default=None,
        help='End frame'
    )
    parser.add_argument(
        '--use_neutral_shape',
        action='store_true',
        help='Use SMPL_neutral (beta=0) instead of optimized shape'
    )

    args = parser.parse_args()

    # Convert paths
    b3d_path = Path(args.b3d)
    smpl_params_path = Path(args.smpl_params)
    smpl_model_path = Path(args.smpl_model)
    rajagopal_mesh_path = Path(args.rajagopal_mesh)
    rajagopal_seg_path = Path(args.rajagopal_seg)
    output_dir = Path(args.output_dir)

    # Validate
    if not b3d_path.exists():
        print(f"Error: .b3d file not found: {b3d_path}")
        sys.exit(1)
    if not smpl_params_path.exists():
        print(f"Error: SMPL params not found: {smpl_params_path}")
        sys.exit(1)
    if not smpl_model_path.exists():
        print(f"Error: SMPL model not found: {smpl_model_path}")
        sys.exit(1)
    if not rajagopal_mesh_path.exists():
        print(f"Error: Rajagopal mesh not found: {rajagopal_mesh_path}")
        sys.exit(1)
    if not rajagopal_seg_path.exists():
        print(f"Error: Bone segmentation not found: {rajagopal_seg_path}")
        sys.exit(1)

    # Frame range
    frame_range = None
    if args.start_frame is not None or args.end_frame is not None:
        start = args.start_frame if args.start_frame is not None else 0
        end = args.end_frame if args.end_frame is not None else float('inf')
        frame_range = (start, int(end))

    # Export
    export_comparison_with_rajagopal(
        b3d_path=b3d_path,
        smpl_params_path=smpl_params_path,
        smpl_model_path=smpl_model_path,
        rajagopal_mesh_path=rajagopal_mesh_path,
        rajagopal_seg_path=rajagopal_seg_path,
        output_dir=output_dir,
        frame_range=frame_range,
        use_neutral_shape=args.use_neutral_shape
    )


if __name__ == '__main__':
    main()
