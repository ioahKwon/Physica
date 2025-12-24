#!/usr/bin/env python3
"""
Visualize how SKEL scapula DOFs affect the mesh and skeleton.

SKEL Scapula DOFs (Right side):
- DOF 26: scapula_abduction_r (protraction/retraction - horizontal movement)
- DOF 27: scapula_elevation_r (elevation/depression - vertical movement)
- DOF 28: scapula_upward_rot_r (upward/downward rotation)

SKEL Scapula DOFs (Left side):
- DOF 36: scapula_abduction_l
- DOF 37: scapula_elevation_l
- DOF 38: scapula_upward_rot_l
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SKEL')

from models.skel_model import SKELModelWrapper, SKEL_JOINT_NAMES, SKEL_NUM_POSE_DOF

SKEL_MODEL_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'


def create_sphere(center, radius=0.02, n_segments=8):
    """Create sphere vertices and faces"""
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


def create_cylinder(start, end, radius=0.008, n_segments=8):
    """Create cylinder between two points"""
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
    for t, center in enumerate([start, end]):
        for i in range(n_segments):
            angle = 2 * np.pi * i / n_segments
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(center + offset)

    faces = []
    for i in range(n_segments):
        p1, p2 = i, (i + 1) % n_segments
        p3, p4 = n_segments + (i + 1) % n_segments, n_segments + i
        faces.append([p1, p2, p3])
        faces.append([p1, p3, p4])

    # Caps
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
    """Create spheres for all joints"""
    all_verts, all_faces = [], []
    offset = 0
    for j in joints:
        v, f = create_sphere(j, radius)
        all_verts.append(v)
        all_faces.append(f + offset)
        offset += len(v)
    return np.vstack(all_verts), np.vstack(all_faces)


def create_skeleton_bones(joints, parents, radius=0.01):
    """Create cylinders for bones"""
    all_verts, all_faces = [], []
    offset = 0
    for i, p in enumerate(parents):
        if p >= 0:
            v, f = create_cylinder(joints[p], joints[i], radius)
            if len(v) > 0:
                all_verts.append(v)
                all_faces.append(f + offset)
                offset += len(v)
    if all_verts:
        return np.vstack(all_verts), np.vstack(all_faces)
    return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)


def save_obj(vertices, faces, filepath):
    """Save OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


def get_skel_parents(skel_model):
    """Get SKEL parent indices from the model"""
    parents = skel_model.parents.cpu().numpy().tolist()
    return parents


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/03_Output/scapula_dof_visualization')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load SKEL model
    skel = SKELModelWrapper(model_path=SKEL_MODEL_PATH, gender='male', device=device)
    parents = get_skel_parents(skel)
    print(f"SKEL parents: {parents}")
    print(f"Joint names: {SKEL_JOINT_NAMES}")

    # DOF configuration with descriptions
    scapula_dofs = {
        'abduction_R': {
            'dof': 26,
            'desc': 'protraction(+) / retraction(-)',
            'positive': 'away_from_spine',
            'negative': 'toward_spine'
        },
        'elevation_R': {
            'dof': 27,
            'desc': 'elevation(+) / depression(-)',
            'positive': 'shrug_up',
            'negative': 'push_down'
        },
        'upward_rot_R': {
            'dof': 28,
            'desc': 'upward(+) / downward(-)',
            'positive': 'glenoid_up',
            'negative': 'glenoid_down'
        },
        'abduction_L': {
            'dof': 36,
            'desc': 'protraction(+) / retraction(-)',
            'positive': 'away_from_spine',
            'negative': 'toward_spine'
        },
        'elevation_L': {
            'dof': 37,
            'desc': 'elevation(+) / depression(-)',
            'positive': 'shrug_up',
            'negative': 'push_down'
        },
        'upward_rot_L': {
            'dof': 38,
            'desc': 'upward(+) / downward(-)',
            'positive': 'glenoid_up',
            'negative': 'glenoid_down'
        },
    }

    # Test values (radians) with descriptive names
    test_values = [
        (-0.5, 'neg0.5rad'),
        (-0.3, 'neg0.3rad'),
        (0.0, 'ZERO'),
        (0.3, 'pos0.3rad'),
        (0.5, 'pos0.5rad'),
    ]

    print("\n=== Generating Scapula DOF Visualizations ===")
    print(f"Output directory: {args.output_dir}")
    print(f"SKEL_NUM_POSE_DOF: {SKEL_NUM_POSE_DOF}")

    # Generate base T-pose first
    print("\n--- Generating T-pose (all zeros) ---")
    betas = torch.zeros(1, 10, device=device)
    poses = torch.zeros(1, SKEL_NUM_POSE_DOF, device=device)
    trans = torch.zeros(1, 3, device=device)

    verts, joints, skel_verts = skel.forward(betas, poses, trans, return_skeleton=True)
    verts = verts[0].cpu().numpy()
    joints = joints[0].cpu().numpy()
    skel_verts = skel_verts[0].cpu().numpy()
    faces = skel.faces
    skel_faces = skel.skel_faces

    # Create 00_reference folder for T-pose
    ref_dir = os.path.join(args.output_dir, '00_reference_Tpose')
    os.makedirs(ref_dir, exist_ok=True)

    save_obj(verts, faces, os.path.join(ref_dir, 'mesh.obj'))
    joint_verts, joint_faces = create_joint_spheres(joints, radius=0.015)
    save_obj(joint_verts, joint_faces, os.path.join(ref_dir, 'joints.obj'))
    bone_verts, bone_faces = create_skeleton_bones(joints, parents, radius=0.008)
    if len(bone_verts) > 0:
        save_obj(bone_verts, bone_faces, os.path.join(ref_dir, 'skeleton.obj'))
    if skel_faces is not None:
        save_obj(skel_verts, skel_faces, os.path.join(ref_dir, 'skeleton_mesh.obj'))
    print(f"  Saved to: {ref_dir}/")

    # Generate variations for each scapula DOF
    for dof_name, dof_info in scapula_dofs.items():
        dof_idx = dof_info['dof']
        dof_desc = dof_info['desc']

        # Create folder for this DOF
        dof_dir = os.path.join(args.output_dir, f'{dof_name}_DOF{dof_idx}')
        os.makedirs(dof_dir, exist_ok=True)

        print(f"\n--- {dof_name} (DOF {dof_idx}): {dof_desc} ---")

        for val, val_name in test_values:
            # Reset pose
            poses = torch.zeros(1, SKEL_NUM_POSE_DOF, device=device)
            poses[0, dof_idx] = val

            verts, joints, skel_verts = skel.forward(betas, poses, trans, return_skeleton=True)
            verts = verts[0].cpu().numpy()
            joints = joints[0].cpu().numpy()
            skel_verts = skel_verts[0].cpu().numpy()

            # Create subfolder for this value
            val_dir = os.path.join(dof_dir, val_name)
            os.makedirs(val_dir, exist_ok=True)

            # Save mesh, joints, skeleton
            save_obj(verts, faces, os.path.join(val_dir, 'mesh.obj'))

            joint_verts, joint_faces = create_joint_spheres(joints, radius=0.015)
            save_obj(joint_verts, joint_faces, os.path.join(val_dir, 'joints.obj'))

            bone_verts, bone_faces = create_skeleton_bones(joints, parents, radius=0.008)
            if len(bone_verts) > 0:
                save_obj(bone_verts, bone_faces, os.path.join(val_dir, 'skeleton.obj'))

            if skel_faces is not None:
                save_obj(skel_verts, skel_faces, os.path.join(val_dir, 'skeleton_mesh.obj'))

            print(f"  {val_name} (val={val:+.1f}rad) -> {val_dir}/")

    # Also generate combined left+right variations
    print("\n--- Combined Left+Right Scapula Variations ---")
    combined_dir = os.path.join(args.output_dir, 'combined_both_sides')
    os.makedirs(combined_dir, exist_ok=True)

    combined_tests = [
        ('abduction_POSITIVE_both', [(26, 0.3), (36, 0.3)], 'protraction'),
        ('abduction_NEGATIVE_both', [(26, -0.3), (36, -0.3)], 'retraction'),
        ('elevation_POSITIVE_both', [(27, 0.3), (37, 0.3)], 'shrug_up'),
        ('elevation_NEGATIVE_both', [(27, -0.3), (37, -0.3)], 'push_down'),
        ('upward_rot_POSITIVE_both', [(28, 0.3), (38, 0.3)], 'glenoid_up'),
        ('upward_rot_NEGATIVE_both', [(28, -0.3), (38, -0.3)], 'glenoid_down'),
    ]

    for name, dof_vals, desc in combined_tests:
        poses = torch.zeros(1, SKEL_NUM_POSE_DOF, device=device)
        for dof_idx, val in dof_vals:
            poses[0, dof_idx] = val

        verts, joints, skel_verts = skel.forward(betas, poses, trans, return_skeleton=True)
        verts = verts[0].cpu().numpy()
        joints = joints[0].cpu().numpy()
        skel_verts = skel_verts[0].cpu().numpy()

        # Create subfolder
        sub_dir = os.path.join(combined_dir, name)
        os.makedirs(sub_dir, exist_ok=True)

        save_obj(verts, faces, os.path.join(sub_dir, 'mesh.obj'))
        joint_verts, joint_faces = create_joint_spheres(joints, radius=0.015)
        save_obj(joint_verts, joint_faces, os.path.join(sub_dir, 'joints.obj'))
        bone_verts, bone_faces = create_skeleton_bones(joints, parents, radius=0.008)
        if len(bone_verts) > 0:
            save_obj(bone_verts, bone_faces, os.path.join(sub_dir, 'skeleton.obj'))
        if skel_faces is not None:
            save_obj(skel_verts, skel_faces, os.path.join(sub_dir, 'skeleton_mesh.obj'))
        print(f"  {name} ({desc}) -> {sub_dir}/")

    print(f"\n=== Done ===")
    print(f"Output directory: {args.output_dir}")
    print("\nDOF Summary:")
    print("  Right scapula: DOF 26 (abduction), 27 (elevation), 28 (upward_rot)")
    print("  Left scapula:  DOF 36 (abduction), 37 (elevation), 38 (upward_rot)")
    print("\nMovement meanings:")
    print("  Abduction (+): Scapula moves away from spine (protraction)")
    print("  Abduction (-): Scapula moves toward spine (retraction)")
    print("  Elevation (+): Scapula moves up (shrugging)")
    print("  Elevation (-): Scapula moves down (depression)")
    print("  Upward_rot (+): Glenoid faces up (arm raising)")
    print("  Upward_rot (-): Glenoid faces down")


if __name__ == '__main__':
    main()
