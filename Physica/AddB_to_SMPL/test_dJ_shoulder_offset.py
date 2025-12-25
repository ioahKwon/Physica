#!/usr/bin/env python3
"""
Use SKEL's dJ parameter to directly offset scapula joints

SKEL forward function accepts dJ: B x 24 x 3 tensor of joint offsets.
We use this to directly push scapula joints outward without modifying templates.

Joint indices:
  14: scapula_r (right shoulder blade)
  15: humerus_r (right upper arm)
  19: scapula_l (left shoulder blade)
  20: humerus_l (left upper arm)
"""

import os
import sys
import numpy as np
import torch

sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/SKEL')
sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

from skel.skel_model import SKEL
from models.smpl_model import SMPLModel
import nimblephysics as nimble

def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """Save mesh as OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def save_skeleton_obj(joints: np.ndarray, kintree: np.ndarray, filepath: str, sphere_radius=0.015):
    """Save skeleton as OBJ with spheres at joints and cylinders for bones"""
    with open(filepath, 'w') as f:
        vertex_offset = 0

        # Draw spheres at each joint
        for j_idx, joint in enumerate(joints):
            # Create a small sphere (icosahedron approximation)
            # 12 vertices for icosahedron
            phi = (1 + np.sqrt(5)) / 2
            sphere_verts = np.array([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ]) * sphere_radius / np.sqrt(1 + phi**2)

            sphere_verts = sphere_verts + joint

            for v in sphere_verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Icosahedron faces
            sphere_faces = [
                [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
            ]

            for face in sphere_faces:
                f.write(f"f {face[0]+1+vertex_offset} {face[1]+1+vertex_offset} {face[2]+1+vertex_offset}\n")

            vertex_offset += 12

        # Draw bones (cylinders as simple lines of triangles)
        num_joints = len(joints)
        for i in range(1, num_joints):
            parent_idx = kintree[0, i]
            if parent_idx >= 0 and parent_idx < num_joints:
                p1 = joints[parent_idx]
                p2 = joints[i]

                # Create cylinder with 8 segments
                n_seg = 8
                bone_radius = 0.008

                # Direction vector
                direction = p2 - p1
                length = np.linalg.norm(direction)
                if length < 1e-6:
                    continue
                direction = direction / length

                # Find perpendicular vectors
                if abs(direction[1]) < 0.9:
                    perp1 = np.cross(direction, [0, 1, 0])
                else:
                    perp1 = np.cross(direction, [1, 0, 0])
                perp1 = perp1 / np.linalg.norm(perp1)
                perp2 = np.cross(direction, perp1)

                # Create cylinder vertices
                cyl_verts = []
                for end_point in [p1, p2]:
                    for seg in range(n_seg):
                        angle = 2 * np.pi * seg / n_seg
                        offset = bone_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                        cyl_verts.append(end_point + offset)

                for v in cyl_verts:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                # Create cylinder faces
                for seg in range(n_seg):
                    next_seg = (seg + 1) % n_seg
                    # Bottom triangle
                    v1 = vertex_offset + seg
                    v2 = vertex_offset + next_seg
                    v3 = vertex_offset + n_seg + seg
                    v4 = vertex_offset + n_seg + next_seg
                    f.write(f"f {v1+1} {v2+1} {v3+1}\n")
                    f.write(f"f {v2+1} {v4+1} {v3+1}\n")

                vertex_offset += 2 * n_seg

def save_addb_skeleton_obj(joints: np.ndarray, parent_indices: list, filepath: str, sphere_radius=0.015):
    """Save AddB skeleton as OBJ with spheres at joints and cylinders for bones"""
    with open(filepath, 'w') as f:
        vertex_offset = 0

        # Draw spheres at each joint
        for j_idx, joint in enumerate(joints):
            phi = (1 + np.sqrt(5)) / 2
            sphere_verts = np.array([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ]) * sphere_radius / np.sqrt(1 + phi**2)

            sphere_verts = sphere_verts + joint

            for v in sphere_verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            sphere_faces = [
                [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
            ]

            for face in sphere_faces:
                f.write(f"f {face[0]+1+vertex_offset} {face[1]+1+vertex_offset} {face[2]+1+vertex_offset}\n")

            vertex_offset += 12

        # Draw bones using parent_indices
        num_joints = len(joints)
        for i in range(num_joints):
            parent_idx = parent_indices[i]
            if parent_idx >= 0 and parent_idx < num_joints:
                p1 = joints[parent_idx]
                p2 = joints[i]

                n_seg = 8
                bone_radius = 0.008

                direction = p2 - p1
                length = np.linalg.norm(direction)
                if length < 1e-6:
                    continue
                direction = direction / length

                if abs(direction[1]) < 0.9:
                    perp1 = np.cross(direction, [0, 1, 0])
                else:
                    perp1 = np.cross(direction, [1, 0, 0])
                perp1 = perp1 / np.linalg.norm(perp1)
                perp2 = np.cross(direction, perp1)

                cyl_verts = []
                for end_point in [p1, p2]:
                    for seg in range(n_seg):
                        angle = 2 * np.pi * seg / n_seg
                        offset = bone_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                        cyl_verts.append(end_point + offset)

                for v in cyl_verts:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                for seg in range(n_seg):
                    next_seg = (seg + 1) % n_seg
                    v1 = vertex_offset + seg
                    v2 = vertex_offset + next_seg
                    v3 = vertex_offset + n_seg + seg
                    v4 = vertex_offset + n_seg + next_seg
                    f.write(f"f {v1+1} {v2+1} {v3+1}\n")
                    f.write(f"f {v2+1} {v4+1} {v3+1}\n")

                vertex_offset += 2 * n_seg

def load_addb_skeleton(b3d_path: str, frame: int = 0):
    """Load AddB skeleton from b3d file"""
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    skel = subject.readSkel(0)

    # Read frame
    frames = subject.readFrames(trial=0, startFrame=frame, numFramesToRead=1, contactThreshold=20)
    pp = frames[0].processingPasses[0]
    pos = np.asarray(pp.pos, dtype=np.float32)
    skel.setPositions(pos)

    # Get joint positions and parent info
    joints = []
    joint_names = []
    parent_indices = []
    joint_name_to_idx = {}

    for i in range(skel.getNumJoints()):
        joint = skel.getJoint(i)
        world = joint.getChildBodyNode().getWorldTransform().translation()
        name = joint.getName()

        joints.append(world)
        joint_names.append(name)
        joint_name_to_idx[name] = i

        # Get parent
        parent_body = joint.getParentBodyNode()
        if parent_body is not None:
            parent_name = None
            for j in range(skel.getNumJoints()):
                if skel.getJoint(j).getChildBodyNode() == parent_body:
                    parent_name = skel.getJoint(j).getName()
                    break
            if parent_name and parent_name in joint_name_to_idx:
                parent_indices.append(joint_name_to_idx[parent_name])
            else:
                parent_indices.append(-1)
        else:
            parent_indices.append(-1)

    return np.array(joints), joint_names, parent_indices

def main():
    device = torch.device('cpu')
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_dJ_offset'
    os.makedirs(out_dir, exist_ok=True)

    # Load params
    skel_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/skel/skel_params.npz')
    smpl_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/smpl/smpl_params.npz')

    betas = torch.from_numpy(skel_params['betas']).float().to(device).unsqueeze(0)
    poses = torch.from_numpy(skel_params['poses'][0:1]).float().to(device)
    trans = torch.from_numpy(skel_params['trans'][0:1]).float().to(device)

    print("=" * 70)
    print("dJ-based Shoulder Widening")
    print("=" * 70)

    # Load SKEL model
    skel = SKEL(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male'
    ).to(device)

    print(f"\nSKEL model loaded")
    print(f"  Joints: {skel.num_joints}")

    # Get original output (no dJ)
    output_orig = skel(
        betas=betas,
        poses=poses,
        trans=trans,
        dJ=None
    )

    verts_orig = output_orig.skin_verts[0].detach().cpu().numpy()
    joints_orig = output_orig.joints[0].detach().cpu().numpy()
    faces = skel.skin_f.cpu().numpy()
    kintree = skel.kintree_table.cpu().numpy()

    save_obj(verts_orig, faces, os.path.join(out_dir, '1_original.obj'))
    save_skeleton_obj(joints_orig, kintree, os.path.join(out_dir, '1_original_skeleton.obj'))

    print(f"\nOriginal joint positions:")
    print(f"  14 (scapula_r): {joints_orig[14]}")
    print(f"  15 (humerus_r): {joints_orig[15]}")
    print(f"  19 (scapula_l): {joints_orig[19]}")
    print(f"  20 (humerus_l): {joints_orig[20]}")

    # Calculate original shoulder width using scapula joints
    scapula_r = joints_orig[14]
    scapula_l = joints_orig[19]
    orig_width = np.abs(scapula_l[0] - scapula_r[0])
    print(f"\nOriginal shoulder width (X distance): {orig_width:.4f}m")

    # Test different offsets using dJ
    offsets = [0.02, 0.05, 0.08, 0.10]

    for offset in offsets:
        print(f"\n[Offset {offset:.2f}m = {int(offset*100)}cm]")

        # Create dJ tensor: B x 24 x 3
        dJ = torch.zeros(1, 24, 3, device=device)

        # Push scapula_r outward (negative X direction for right side)
        dJ[0, 14, 0] = -offset  # scapula_r X offset

        # Push scapula_l outward (positive X direction for left side)
        dJ[0, 19, 0] = +offset  # scapula_l X offset

        # Also offset the connected joints (humerus) to avoid disconnection
        dJ[0, 15, 0] = -offset  # humerus_r follows scapula_r
        dJ[0, 20, 0] = +offset  # humerus_l follows scapula_l

        # Forward pass with dJ
        output_mod = skel(
            betas=betas,
            poses=poses,
            trans=trans,
            dJ=dJ
        )

        verts_mod = output_mod.skin_verts[0].detach().cpu().numpy()
        joints_mod = output_mod.joints[0].detach().cpu().numpy()

        save_obj(verts_mod, faces, os.path.join(out_dir, f'2_dJ_offset_{int(offset*100)}cm.obj'))
        save_skeleton_obj(joints_mod, kintree, os.path.join(out_dir, f'2_dJ_offset_{int(offset*100)}cm_skeleton.obj'))

        # Check new shoulder width
        new_scapula_r = joints_mod[14]
        new_scapula_l = joints_mod[19]
        new_width = np.abs(new_scapula_l[0] - new_scapula_r[0])

        print(f"  New scapula_r: {new_scapula_r}")
        print(f"  New scapula_l: {new_scapula_l}")
        print(f"  New shoulder width: {new_width:.4f}m (was {orig_width:.4f}m)")
        print(f"  Width increase: {(new_width - orig_width)*100:.2f}cm")

    # Save SMPL for reference
    print("\n[Reference] SMPL...")
    smpl = SMPLModel(gender='male', device=device)
    smpl_beta = torch.from_numpy(smpl_params['betas']).float().to(device)
    smpl_poses = torch.from_numpy(smpl_params['poses']).float().to(device)
    smpl_trans = torch.from_numpy(smpl_params['trans']).float().to(device)

    smpl_verts, _ = smpl.forward(
        betas=smpl_beta.unsqueeze(0),
        poses=smpl_poses[0:1],
        trans=smpl_trans[0:1]
    )
    save_obj(smpl_verts[0].detach().cpu().numpy(), smpl.faces.cpu().numpy(),
             os.path.join(out_dir, '3_smpl.obj'))

    # Load and save AddB skeleton
    print("\n[Reference] AddB skeleton...")
    b3d_path = '/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject1/Subject1.b3d'
    addb_joints, addb_names, addb_parents = load_addb_skeleton(b3d_path, frame=0)
    save_addb_skeleton_obj(addb_joints, addb_parents, os.path.join(out_dir, '4_addb_skeleton.obj'))
    print(f"  AddB joints: {len(addb_joints)}")

    print("\n" + "=" * 70)
    print("Files saved to:")
    print(f"  {out_dir}")
    print("\nFiles:")
    print("  1_original.obj + 1_original_skeleton.obj - SKEL original")
    print("  2_dJ_offset_Xcm.obj + 2_dJ_offset_Xcm_skeleton.obj - SKEL widened")
    print("  3_smpl.obj - SMPL reference")
    print("  4_addb_skeleton.obj - AddB skeleton (ground truth)")
    print("\nUsing SKEL's dJ parameter to directly offset scapula joints.")

if __name__ == '__main__':
    main()
