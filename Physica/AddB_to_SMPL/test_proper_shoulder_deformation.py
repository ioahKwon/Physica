#!/usr/bin/env python3
"""
Test Proper Shoulder Deformation

Purpose: Move shoulder joints AND deform vertices using LBS weights
"""

import os
import sys
import pickle
import numpy as np
import torch

sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/SKEL')
sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

from models.skel_model import SKELModelWrapper
from models.smpl_model import SMPLModel

def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """Save mesh as OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def create_sphere(center, radius=0.02, segments=8):
    """Create sphere mesh"""
    vertices = []
    faces = []

    for i in range(segments + 1):
        lat = np.pi * i / segments - np.pi / 2
        for j in range(segments):
            lon = 2 * np.pi * j / segments
            x = radius * np.cos(lat) * np.cos(lon) + center[0]
            y = radius * np.cos(lat) * np.sin(lon) + center[1]
            z = radius * np.sin(lat) + center[2]
            vertices.append([x, y, z])

    for i in range(segments):
        for j in range(segments):
            p1 = i * segments + j
            p2 = i * segments + (j + 1) % segments
            p3 = (i + 1) * segments + (j + 1) % segments
            p4 = (i + 1) * segments + j
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])

    return np.array(vertices), np.array(faces)

def create_cylinder(start, end, radius=0.01, segments=8):
    """Create cylinder mesh between two points"""
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
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(center + offset)

    for i in range(segments):
        p1 = i
        p2 = (i + 1) % segments
        p3 = segments + (i + 1) % segments
        p4 = segments + i
        faces.append([p1, p2, p3])
        faces.append([p1, p3, p4])

    return np.array(vertices), np.array(faces)

def save_skeleton(joints, parent_indices, filepath, joint_radius=0.02, bone_radius=0.008):
    """Save skeleton as OBJ"""
    all_verts = []
    all_faces = []
    vert_offset = 0

    # Joint spheres
    for joint in joints:
        verts, faces = create_sphere(joint, radius=joint_radius)
        all_verts.append(verts)
        all_faces.append(faces + vert_offset)
        vert_offset += len(verts)

    # Bone cylinders
    for j_idx, parent_idx in enumerate(parent_indices):
        if parent_idx >= 0:
            start = joints[parent_idx]
            end = joints[j_idx]
            verts, faces = create_cylinder(start, end, radius=bone_radius)
            if len(verts) > 0:
                all_verts.append(verts)
                all_faces.append(faces + vert_offset)
                vert_offset += len(verts)

    if len(all_verts) == 0:
        return

    all_verts = np.concatenate(all_verts, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)
    save_obj(all_verts, all_faces, filepath)

def deform_mesh_with_joint_offset(vertices, joints_original, joints_modified, weights,
                                   shoulder_r_idx, shoulder_l_idx):
    """
    Deform mesh based on joint movement using skinning weights

    Simple approach: Move vertices proportionally to how much they're influenced by shoulder joints
    """
    verts_deformed = vertices.copy()

    # Calculate joint displacement
    joint_offset_r = joints_modified[shoulder_r_idx] - joints_original[shoulder_r_idx]
    joint_offset_l = joints_modified[shoulder_l_idx] - joints_original[shoulder_l_idx]

    # Apply displacement to vertices based on skinning weights
    for v_idx in range(len(verts_deformed)):
        w_r = weights[v_idx, shoulder_r_idx]
        w_l = weights[v_idx, shoulder_l_idx]

        # Move vertex proportional to its skinning weights
        if w_r > 0.001:
            verts_deformed[v_idx] += joint_offset_r * w_r

        if w_l > 0.001:
            verts_deformed[v_idx] += joint_offset_l * w_l

    return verts_deformed

def main():
    device = torch.device('cpu')
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_proper_deform'
    os.makedirs(out_dir, exist_ok=True)

    # Load SKEL pkl for weights
    skel_pkl = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1/skel_male.pkl'
    with open(skel_pkl, 'rb') as f:
        skel_data = pickle.load(f, encoding='latin1')

    weights = skel_data['skin_weights'].toarray()  # [6890, 24]

    # Load parameters
    skel_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/skel/skel_params.npz')
    smpl_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/smpl/smpl_params.npz')

    skel_beta = torch.from_numpy(skel_params['betas']).float().to(device)
    skel_poses = torch.from_numpy(skel_params['poses']).float().to(device)
    skel_trans = torch.from_numpy(skel_params['trans']).float().to(device)

    smpl_beta = torch.from_numpy(smpl_params['betas']).float().to(device)
    smpl_poses = torch.from_numpy(smpl_params['poses']).float().to(device)
    smpl_trans = torch.from_numpy(smpl_params['trans']).float().to(device)

    frame = 0

    print("=" * 70)
    print("Proper Shoulder Deformation Test")
    print("=" * 70)

    # SKEL parents
    skel_parents = np.array([
        -1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 11, 12,
        14, 14, 16, 17,
        15, 15, 19, 20
    ])

    # Initialize SKEL
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    # Get original mesh
    print("\n[1] Original SKEL...")
    skel_verts_orig, skel_joints_orig = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_poses[frame:frame+1],
        trans=skel_trans[frame:frame+1]
    )

    verts_np = skel_verts_orig[0].detach().cpu().numpy()
    joints_np = skel_joints_orig[0].detach().cpu().numpy()

    save_obj(verts_np, skel.faces, os.path.join(out_dir, '1_skel_mesh_original.obj'))
    save_skeleton(joints_np, skel_parents, os.path.join(out_dir, '1_skel_skeleton_original.obj'))

    # Joint indices in SKEL wrapper output
    right_shoulder_idx_wrapper = 15  # humerus_r in wrapper
    left_shoulder_idx_wrapper = 20   # humerus_l in wrapper

    # Joint indices in pkl weights
    right_shoulder_idx_pkl = 17  # right_shoulder in pkl
    left_shoulder_idx_pkl = 16   # left_shoulder in pkl

    orig_width = np.linalg.norm(joints_np[right_shoulder_idx_wrapper] - joints_np[left_shoulder_idx_wrapper])
    print(f"  Original shoulder width: {orig_width:.4f}m")

    # Test different offsets
    offsets = [0.05, 0.10, 0.15, 0.20]

    for offset in offsets:
        print(f"\n[Offset ±{offset:.2f}m = ±{int(offset*100)}cm]")

        # Modify joint positions
        joints_modified = joints_np.copy()
        joints_modified[right_shoulder_idx_wrapper, 0] -= offset  # Move right shoulder outward
        joints_modified[left_shoulder_idx_wrapper, 0] += offset   # Move left shoulder outward

        # Deform mesh based on joint movement
        verts_deformed = deform_mesh_with_joint_offset(
            verts_np,
            joints_np,
            joints_modified,
            weights,
            right_shoulder_idx_pkl,  # Use pkl indices for weights
            left_shoulder_idx_pkl
        )

        # Save deformed mesh
        save_obj(verts_deformed, skel.faces,
                 os.path.join(out_dir, f'2_skel_mesh_offset_{int(offset*100)}cm.obj'))

        # Save skeleton
        save_skeleton(joints_modified, skel_parents,
                     os.path.join(out_dir, f'2_skel_skeleton_offset_{int(offset*100)}cm.obj'))

        new_width = np.linalg.norm(joints_modified[right_shoulder_idx_wrapper] - joints_modified[left_shoulder_idx_wrapper])
        width_increase = (new_width - orig_width) / orig_width * 100
        print(f"  New shoulder width: {new_width:.4f}m ({width_increase:+.1f}%)")

    # SMPL reference
    print("\n[Reference] SMPL...")
    smpl = SMPLModel(gender='male', device=device)
    smpl_verts, smpl_joints = smpl.forward(
        betas=smpl_beta.unsqueeze(0),
        poses=smpl_poses[frame:frame+1],
        trans=smpl_trans[frame:frame+1]
    )

    save_obj(smpl_verts[0].detach().cpu().numpy(), smpl.faces.cpu().numpy(),
             os.path.join(out_dir, '3_smpl_mesh.obj'))

    smpl_parents = np.array([
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
        16, 17, 18, 19, 20, 21
    ])

    smpl_joints_np = smpl_joints[0].detach().cpu().numpy()
    save_skeleton(smpl_joints_np, smpl_parents, os.path.join(out_dir, '3_smpl_skeleton.obj'))

    smpl_width = np.linalg.norm(smpl_joints_np[17] - smpl_joints_np[16])
    print(f"  SMPL shoulder width: {smpl_width:.4f}m")

    print("\n" + "=" * 70)
    print("Complete! Files saved to:")
    print(f"  {out_dir}")
    print("\nNow both mesh AND skeleton show the deformation!")

if __name__ == '__main__':
    main()
