"""
Visualization utilities for creating OBJ meshes from joints and skeletons.
"""

from typing import List, Tuple, Optional
import numpy as np


def create_sphere(
    center: np.ndarray,
    radius: float = 0.02,
    n_segments: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sphere vertices and faces.

    Args:
        center: Sphere center [3].
        radius: Sphere radius.
        n_segments: Number of segments (resolution).

    Returns:
        vertices: [V, 3]
        faces: [F, 3] (0-indexed)
    """
    vertices = []
    for i in range(n_segments + 1):
        lat = np.pi * i / n_segments - np.pi / 2
        for j in range(n_segments):
            lon = 2 * np.pi * j / n_segments
            x = radius * np.cos(lat) * np.cos(lon) + center[0]
            y = radius * np.cos(lat) * np.sin(lon) + center[1]
            z = radius * np.sin(lat) + center[2]
            vertices.append([x, y, z])

    faces = []
    for i in range(n_segments):
        for j in range(n_segments):
            p1 = i * n_segments + j
            p2 = i * n_segments + (j + 1) % n_segments
            p3 = (i + 1) * n_segments + (j + 1) % n_segments
            p4 = (i + 1) * n_segments + j
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

    return np.array(vertices), np.array(faces)


def create_cylinder(
    start: np.ndarray,
    end: np.ndarray,
    radius: float = 0.008,
    n_segments: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create cylinder between two points.

    Args:
        start: Start point [3].
        end: End point [3].
        radius: Cylinder radius.
        n_segments: Number of segments (resolution).

    Returns:
        vertices: [V, 3]
        faces: [F, 3] (0-indexed)
    """
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    direction = direction / length

    # Find perpendicular vectors
    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(direction, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    # Create vertices
    vertices = []
    for t, center in enumerate([start, end]):
        for i in range(n_segments):
            angle = 2 * np.pi * i / n_segments
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(center + offset)

    # Create side faces
    faces = []
    for i in range(n_segments):
        p1, p2 = i, (i + 1) % n_segments
        p3, p4 = n_segments + (i + 1) % n_segments, n_segments + i
        faces.append([p1, p2, p3])
        faces.append([p1, p3, p4])

    # Create caps
    start_center_idx = len(vertices)
    vertices.append(start)
    for i in range(n_segments):
        faces.append([start_center_idx, (i + 1) % n_segments, i])

    end_center_idx = len(vertices)
    vertices.append(end)
    for i in range(n_segments):
        faces.append([end_center_idx, n_segments + i, n_segments + (i + 1) % n_segments])

    return np.array(vertices), np.array(faces)


def create_joint_spheres(
    joints: np.ndarray,
    radius: float = 0.015,
    n_segments: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create spheres for all joints.

    Args:
        joints: Joint positions [J, 3].
        radius: Sphere radius.
        n_segments: Resolution.

    Returns:
        vertices: Combined vertices [V, 3]
        faces: Combined faces [F, 3]
    """
    all_verts = []
    all_faces = []
    offset = 0

    for j in joints:
        v, f = create_sphere(j, radius, n_segments)
        all_verts.append(v)
        all_faces.append(f + offset)
        offset += len(v)

    if all_verts:
        return np.vstack(all_verts), np.vstack(all_faces)
    return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)


def create_skeleton_bones(
    joints: np.ndarray,
    parents: List[int],
    radius: float = 0.008,
    n_segments: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create bone cylinders for skeleton.

    Args:
        joints: Joint positions [J, 3].
        parents: Parent index for each joint (-1 for root).
        radius: Cylinder radius.
        n_segments: Resolution.

    Returns:
        vertices: Combined vertices [V, 3]
        faces: Combined faces [F, 3]
    """
    all_verts = []
    all_faces = []
    offset = 0

    for i, p in enumerate(parents):
        if p >= 0:
            v, f = create_cylinder(joints[p], joints[i], radius, n_segments)
            if len(v) > 0:
                all_verts.append(v)
                all_faces.append(f + offset)
                offset += len(v)

    if all_verts:
        return np.vstack(all_verts), np.vstack(all_faces)
    return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)


def joints_to_obj(
    joints: np.ndarray,
    output_path: str,
    parents: Optional[List[int]] = None,
    joint_radius: float = 0.015,
    bone_radius: float = 0.008,
    include_bones: bool = True,
):
    """
    Save joints (and optionally bones) as OBJ file.

    Args:
        joints: Joint positions [J, 3].
        output_path: Output OBJ path.
        parents: Parent indices for bone connections.
        joint_radius: Radius for joint spheres.
        bone_radius: Radius for bone cylinders.
        include_bones: Whether to include bone connections.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create joint spheres
    joint_verts, joint_faces = create_joint_spheres(joints, joint_radius)

    # Create bones if requested
    if include_bones and parents is not None:
        bone_verts, bone_faces = create_skeleton_bones(joints, parents, bone_radius)
        if len(bone_verts) > 0:
            # Offset bone faces
            bone_faces = bone_faces + len(joint_verts)
            all_verts = np.vstack([joint_verts, bone_verts])
            all_faces = np.vstack([joint_faces, bone_faces])
        else:
            all_verts = joint_verts
            all_faces = joint_faces
    else:
        all_verts = joint_verts
        all_faces = joint_faces

    # Write OBJ
    with open(output_path, 'w') as f:
        for v in all_verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in all_faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


def save_comparison_obj(
    addb_joints: np.ndarray,
    skel_joints: np.ndarray,
    output_dir: str,
    addb_parents: Optional[List[int]] = None,
    skel_parents: Optional[List[int]] = None,
):
    """
    Save AddB and SKEL joints as separate OBJ files for comparison.

    Args:
        addb_joints: AddB joint positions [J_addb, 3].
        skel_joints: SKEL joint positions [J_skel, 3].
        output_dir: Output directory.
        addb_parents: AddB parent indices.
        skel_parents: SKEL parent indices.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    joints_to_obj(
        addb_joints,
        os.path.join(output_dir, 'addb_skeleton.obj'),
        parents=addb_parents,
        joint_radius=0.02,
        bone_radius=0.01,
    )

    joints_to_obj(
        skel_joints,
        os.path.join(output_dir, 'skel_skeleton.obj'),
        parents=skel_parents,
        joint_radius=0.015,
        bone_radius=0.008,
    )
