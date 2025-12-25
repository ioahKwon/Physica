#!/usr/bin/env python3
"""
Generate T-pose comparison between SKEL and AddB (OpenSIM)
- SKEL: mesh + joints + skeleton at origin
- AddB: joints + skeleton with X offset (side by side)

Usage:
    python generate_tpose_comparison.py --skel   # Generate SKEL only
    python generate_tpose_comparison.py --addb   # Generate AddB only
    python generate_tpose_comparison.py --all    # Generate both (default)
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/output_12_15/tpose_comparison'
SKEL_MODEL_DIR = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'
B3D_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject1/Subject1.b3d'

# X offset for AddB (to place side by side)
ADDB_X_OFFSET = 0.0  # No offset - overlay on same position

# SKEL parent indices
SKEL_PARENTS = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 12, 14, 15, 16, 17, 12, 19, 20, 21, 22]


# ---------------------------------------------------------------------------
# OBJ Utilities
# ---------------------------------------------------------------------------

def save_obj(vertices, faces, filepath):
    """Save mesh as OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def create_sphere(center, radius=0.02, n_segments=8):
    """Create a UV sphere mesh at the given center"""
    vertices = []
    faces = []

    for i in range(n_segments + 1):
        lat = np.pi * i / n_segments - np.pi / 2
        for j in range(n_segments):
            lon = 2 * np.pi * j / n_segments
            x = radius * np.cos(lat) * np.cos(lon) + center[0]
            y = radius * np.cos(lat) * np.sin(lon) + center[1]
            z = radius * np.sin(lat) + center[2]
            vertices.append([x, y, z])

    for i in range(n_segments):
        for j in range(n_segments):
            p1 = i * n_segments + j
            p2 = i * n_segments + (j + 1) % n_segments
            p3 = (i + 1) * n_segments + (j + 1) % n_segments
            p4 = (i + 1) * n_segments + j
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

    return np.array(vertices), np.array(faces)


def create_cylinder(start, end, radius=0.008, n_segments=8):
    """Create a cylinder mesh between two points"""
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    direction = direction / length

    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(direction, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    vertices = []
    faces = []

    for t, center in enumerate([start, end]):
        for i in range(n_segments):
            angle = 2 * np.pi * i / n_segments
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(center + offset)

    for i in range(n_segments):
        p1 = i
        p2 = (i + 1) % n_segments
        p3 = n_segments + (i + 1) % n_segments
        p4 = n_segments + i
        faces.append([p1, p2, p3])
        faces.append([p1, p3, p4])

    start_center_idx = len(vertices)
    vertices.append(start)
    for i in range(n_segments):
        faces.append([start_center_idx, (i + 1) % n_segments, i])

    end_center_idx = len(vertices)
    vertices.append(end)
    for i in range(n_segments):
        faces.append([end_center_idx, n_segments + i, n_segments + (i + 1) % n_segments])

    return np.array(vertices), np.array(faces)


def create_joint_spheres(joints, radius=0.02):
    """Create spheres at all joint positions"""
    all_verts = []
    all_faces = []
    vert_offset = 0

    for joint in joints:
        verts, faces = create_sphere(joint, radius)
        all_verts.append(verts)
        all_faces.append(faces + vert_offset)
        vert_offset += len(verts)

    if len(all_verts) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    return np.vstack(all_verts), np.vstack(all_faces)


def create_skeleton_bones(joints, parents, radius=0.008):
    """Create cylinder bones connecting joints based on parent hierarchy"""
    all_verts = []
    all_faces = []
    vert_offset = 0

    for i, parent in enumerate(parents):
        if parent >= 0:
            verts, faces = create_cylinder(joints[parent], joints[i], radius)
            if len(verts) > 0:
                all_verts.append(verts)
                all_faces.append(faces + vert_offset)
                vert_offset += len(verts)

    if len(all_verts) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    return np.vstack(all_verts), np.vstack(all_faces)


# ---------------------------------------------------------------------------
# SKEL Generation
# ---------------------------------------------------------------------------

def generate_skel_tpose():
    """Generate SKEL T-pose mesh, joints, and skeleton"""
    import torch
    from models.skel_model import SKELModelWrapper, SKEL_JOINT_NAMES

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n=== Generating SKEL T-pose ===")

    skel_model = SKELModelWrapper(model_path=SKEL_MODEL_DIR, gender='male', device=device)

    # Zero pose, zero beta
    betas = torch.zeros(10, device=device)
    poses = torch.zeros(46, device=device)
    trans = torch.zeros(3, device=device)

    skel_verts, skel_joints = skel_model.forward(betas, poses, trans)
    skel_verts = skel_verts.cpu().numpy()
    skel_joints = skel_joints.cpu().numpy()
    skel_faces = skel_model.faces

    print(f"SKEL vertices: {skel_verts.shape}")
    print(f"SKEL joints: {skel_joints.shape}")

    # Save SKEL mesh
    save_obj(skel_verts, skel_faces, os.path.join(OUTPUT_DIR, 'skel_tpose_mesh.obj'))
    print(f"Saved: skel_tpose_mesh.obj")

    # Save SKEL joints
    joint_verts, joint_faces = create_joint_spheres(skel_joints, radius=0.02)
    save_obj(joint_verts, joint_faces, os.path.join(OUTPUT_DIR, 'skel_tpose_joints.obj'))
    print(f"Saved: skel_tpose_joints.obj")

    # Save SKEL skeleton
    skel_bone_verts, skel_bone_faces = create_skeleton_bones(skel_joints, SKEL_PARENTS, radius=0.01)
    if len(skel_bone_verts) > 0:
        save_obj(skel_bone_verts, skel_bone_faces, os.path.join(OUTPUT_DIR, 'skel_tpose_skeleton.obj'))
        print(f"Saved: skel_tpose_skeleton.obj")

    # Print SKEL joint positions
    print("\nSKEL joint positions:")
    for i, name in enumerate(SKEL_JOINT_NAMES):
        print(f"  {i:2d}: {name:20s} = [{skel_joints[i, 0]:8.4f}, {skel_joints[i, 1]:8.4f}, {skel_joints[i, 2]:8.4f}]")

    # Save joint positions to numpy
    np.save(os.path.join(OUTPUT_DIR, 'skel_tpose_joints.npy'), skel_joints)
    print(f"Saved: skel_tpose_joints.npy")


# ---------------------------------------------------------------------------
# AddB Generation
# ---------------------------------------------------------------------------

def generate_addb_tpose():
    """Generate AddB T-pose joints and skeleton"""
    import nimblephysics as nimble
    import math

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== Generating AddB T-pose ===")

    # Load B3D
    subject = nimble.biomechanics.SubjectOnDisk(B3D_PATH)
    addb_skel = subject.readSkel(0)

    # Set T-pose: AddB default has arms down, need shoulder abduction ~90 deg
    # DOF 24: acromial_r elevation (abduction to side)
    # DOF 31: acromial_l elevation (abduction to side)
    num_dofs = addb_skel.getNumDofs()
    tpose = [0.0] * num_dofs
    tpose[24] = math.radians(-90)  # Right shoulder abduction (arm to side)
    tpose[31] = math.radians(-90)  # Left shoulder abduction (arm to side)
    addb_skel.setPositions(tpose)
    print(f"Applied T-pose: DOF 24 = -90deg, DOF 31 = -90deg (arms to sides)")

    # Get joint positions
    num_joints = addb_skel.getNumJoints()
    addb_joints = np.zeros((num_joints, 3))
    addb_joint_names = []
    addb_parents = []

    for i in range(num_joints):
        joint = addb_skel.getJoint(i)
        name = joint.getName()
        addb_joint_names.append(name)

        # Get world position
        child_body = joint.getChildBodyNode()
        world_pos = child_body.getWorldTransform().translation()
        addb_joints[i] = world_pos

        # Get parent index
        parent_body = joint.getParentBodyNode()
        if parent_body is None:
            addb_parents.append(-1)
        else:
            # Find parent joint index
            parent_idx = -1
            for j in range(num_joints):
                if addb_skel.getJoint(j).getChildBodyNode() == parent_body:
                    parent_idx = j
                    break
            addb_parents.append(parent_idx)

    print(f"AddB joints: {addb_joints.shape}")
    print(f"AddB joint names: {addb_joint_names}")

    # Print AddB joint positions (original)
    print("\nAddB joint positions (original AddB coordinates):")
    for i, name in enumerate(addb_joint_names):
        print(f"  {i:2d}: {name:20s} = [{addb_joints[i, 0]:8.4f}, {addb_joints[i, 1]:8.4f}, {addb_joints[i, 2]:8.4f}]")

    # Convert AddB coordinates to SKEL coordinates:
    # AddB: X=forward, Y=up, Z=right
    # SKEL: X=right, Y=up, Z=forward
    # Conversion: SKEL_X = -AddB_Z, SKEL_Y = AddB_Y, SKEL_Z = AddB_X
    addb_joints_converted = np.zeros_like(addb_joints)
    addb_joints_converted[:, 0] = -addb_joints[:, 2]  # SKEL_X = -AddB_Z
    addb_joints_converted[:, 1] = addb_joints[:, 1]   # SKEL_Y = AddB_Y
    addb_joints_converted[:, 2] = addb_joints[:, 0]   # SKEL_Z = AddB_X

    # Load SKEL pelvis position for alignment
    skel_joints_path = os.path.join(OUTPUT_DIR, 'skel_tpose_joints.npy')
    if os.path.exists(skel_joints_path):
        skel_joints = np.load(skel_joints_path)
        skel_pelvis = skel_joints[0]  # pelvis is joint 0 in SKEL
        addb_pelvis = addb_joints_converted[0]  # ground_pelvis is joint 0 in AddB

        # Align AddB pelvis to SKEL pelvis
        pelvis_offset = skel_pelvis - addb_pelvis
        addb_joints_converted += pelvis_offset
        print(f"\nAligned to SKEL pelvis: offset = [{pelvis_offset[0]:.4f}, {pelvis_offset[1]:.4f}, {pelvis_offset[2]:.4f}]")
    else:
        print("\nWarning: SKEL joints not found, run with --skel first for alignment")

    print("\nAddB joint positions (converted & aligned to SKEL):")
    for i, name in enumerate(addb_joint_names):
        print(f"  {i:2d}: {name:20s} = [{addb_joints_converted[i, 0]:8.4f}, {addb_joints_converted[i, 1]:8.4f}, {addb_joints_converted[i, 2]:8.4f}]")

    # Apply X offset if needed
    addb_joints_offset = addb_joints_converted.copy()
    addb_joints_offset[:, 0] += ADDB_X_OFFSET

    # Save AddB joints
    addb_joint_verts, addb_joint_faces = create_joint_spheres(addb_joints_offset, radius=0.02)
    save_obj(addb_joint_verts, addb_joint_faces, os.path.join(OUTPUT_DIR, 'addb_tpose_joints.obj'))
    print(f"\nSaved: addb_tpose_joints.obj (X offset = {ADDB_X_OFFSET}m)")

    # Save AddB skeleton
    addb_bone_verts, addb_bone_faces = create_skeleton_bones(addb_joints_offset, addb_parents, radius=0.01)
    if len(addb_bone_verts) > 0:
        save_obj(addb_bone_verts, addb_bone_faces, os.path.join(OUTPUT_DIR, 'addb_tpose_skeleton.obj'))
        print(f"Saved: addb_tpose_skeleton.obj (X offset = {ADDB_X_OFFSET}m)")

    # Save joint positions and names
    np.save(os.path.join(OUTPUT_DIR, 'addb_tpose_joints.npy'), addb_joints)
    np.save(os.path.join(OUTPUT_DIR, 'addb_tpose_joint_names.npy'), np.array(addb_joint_names))
    np.save(os.path.join(OUTPUT_DIR, 'addb_tpose_parents.npy'), np.array(addb_parents))
    print(f"Saved: addb_tpose_joints.npy, addb_tpose_joint_names.npy, addb_tpose_parents.npy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate T-pose comparison')
    parser.add_argument('--skel', action='store_true', help='Generate SKEL only')
    parser.add_argument('--addb', action='store_true', help='Generate AddB only')
    parser.add_argument('--all', action='store_true', help='Generate both (default)')
    args = parser.parse_args()

    # Default to all if no specific option given
    if not args.skel and not args.addb:
        args.all = True

    if args.skel or args.all:
        generate_skel_tpose()

    if args.addb or args.all:
        generate_addb_tpose()

    # Summary
    print(f"\n=== Summary ===")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Load all .obj files in MeshLab to view side by side comparison.")


if __name__ == '__main__':
    main()
