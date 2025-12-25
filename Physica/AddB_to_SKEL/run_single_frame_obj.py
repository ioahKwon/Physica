#!/usr/bin/env python3
"""
Run optimization on Subject11 frame 0 and export OBJ files:
- AddB GT joints + skeleton
- SKEL optimized mesh + joints + skeleton
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SKEL')

from optimize_with_shoulder import optimize_skel_with_shoulder
from models.skel_model import SKEL_JOINT_NAMES
from shoulder_correction import compute_virtual_acromial, ACROMIAL_VERTEX_IDX


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


def save_obj(vertices, faces, filepath, color=None):
    """Save OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            if color:
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n')
            else:
                f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')


def load_addb_frame(b3d_path, trial_idx=0, frame_idx=0):
    """Load a specific frame from AddB b3d file"""
    import nimblephysics as nimble

    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    skel = subject.readSkel(0)

    num_trials = subject.getNumTrials()
    print(f"Number of trials: {num_trials}")
    print(f"Loading trial {trial_idx}, frame {frame_idx}")

    # Read first frame
    frames = subject.readFrames(trial_idx, frame_idx, 1)
    if len(frames) == 0:
        raise ValueError("No frames found")

    frame = frames[0]
    # Position is in processingPasses
    pos = frame.processingPasses[0].pos
    skel.setPositions(pos)

    # Get joint positions
    num_joints = skel.getNumJoints()
    joints = np.zeros((num_joints, 3))
    joint_names = []
    parents = []

    for i in range(num_joints):
        joint = skel.getJoint(i)
        joint_names.append(joint.getName())
        child_body = joint.getChildBodyNode()
        world_pos = child_body.getWorldTransform().translation()
        joints[i] = world_pos

        parent_body = joint.getParentBodyNode()
        if parent_body is None:
            parents.append(-1)
        else:
            parent_idx = -1
            for j in range(num_joints):
                if skel.getJoint(j).getChildBodyNode() == parent_body:
                    parent_idx = j
                    break
            parents.append(parent_idx)

    return joints, joint_names, parents


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default='Subject11', help='Subject name (e.g., Subject1, Subject11)')
    parser.add_argument('--frame', type=int, default=0, help='Frame index')
    parser.add_argument('--trial', type=int, default=0, help='Trial index')
    parser.add_argument('--iters', type=int, default=300, help='Optimization iterations')
    args = parser.parse_args()

    # Paths
    subject_name = args.subject
    b3d_path = f'/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/{subject_name}/{subject_name}.b3d'
    output_dir = f'/egr/research-zijunlab/kwonjoon/03_Output/{subject_name.lower()}_frame{args.frame}_obj'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # =========================================================================
    # 1. Load AddB frame 0
    # =========================================================================
    print("\n=== Loading AddB Frame 0 ===")
    addb_joints, addb_names, addb_parents = load_addb_frame(b3d_path, args.trial, args.frame)
    print(f"AddB joints: {addb_joints.shape}")
    print(f"Joint names: {addb_names}")

    # Convert AddB coordinates to SKEL coordinates
    # AddB: X=forward, Y=up, Z=right
    # SKEL: X=right, Y=up, Z=forward
    addb_joints_converted = np.zeros_like(addb_joints)
    addb_joints_converted[:, 0] = -addb_joints[:, 2]  # SKEL_X = -AddB_Z
    addb_joints_converted[:, 1] = addb_joints[:, 1]   # SKEL_Y = AddB_Y
    addb_joints_converted[:, 2] = addb_joints[:, 0]   # SKEL_Z = AddB_X

    # Save AddB GT joints OBJ
    addb_joint_verts, addb_joint_faces = create_joint_spheres(addb_joints_converted, radius=0.02)
    save_obj(addb_joint_verts, addb_joint_faces,
             os.path.join(output_dir, 'addb_gt_joints.obj'))
    print(f"Saved: addb_gt_joints.obj")

    # Save AddB GT skeleton OBJ
    addb_skel_verts, addb_skel_faces = create_skeleton_bones(addb_joints_converted, addb_parents, radius=0.01)
    if len(addb_skel_verts) > 0:
        save_obj(addb_skel_verts, addb_skel_faces,
                 os.path.join(output_dir, 'addb_gt_skeleton.obj'))
        print(f"Saved: addb_gt_skeleton.obj")

    # =========================================================================
    # 2. Run SKEL optimization
    # =========================================================================
    print("\n=== Running SKEL Optimization ===")

    # Prepare input (single frame)
    target_joints = addb_joints_converted[np.newaxis, :, :]  # [1, 20, 3]

    result = optimize_skel_with_shoulder(
        target_joints=target_joints,
        addb_joint_names=addb_names,
        device=device,
        num_iters=args.iters,
        shoulder_weight=1.0,
        width_weight=0.5,
        verbose=True
    )

    # =========================================================================
    # 3. Save SKEL results
    # =========================================================================
    print("\n=== Saving SKEL Results ===")

    # SKEL mesh (frame 0)
    skel_verts = result['vertices'][0]  # [6890, 3]
    skel_faces = result['faces']
    save_obj(skel_verts, skel_faces, os.path.join(output_dir, 'skel_mesh.obj'))
    print(f"Saved: skel_mesh.obj")

    # SKEL skeleton mesh
    if 'skel_vertices' in result:
        skel_skel_verts = result['skel_vertices'][0]
        skel_skel_faces = result['skel_faces']
        save_obj(skel_skel_verts, skel_skel_faces,
                 os.path.join(output_dir, 'skel_skeleton_mesh.obj'))
        print(f"Saved: skel_skeleton_mesh.obj")

    # SKEL joints
    skel_joints = result['joints'][0]  # [24, 3]
    skel_joint_verts, skel_joint_faces = create_joint_spheres(skel_joints, radius=0.02)
    save_obj(skel_joint_verts, skel_joint_faces,
             os.path.join(output_dir, 'skel_joints.obj'))
    print(f"Saved: skel_joints.obj")

    # SKEL skeleton bones
    skel_parents = result['parents']
    skel_bone_verts, skel_bone_faces = create_skeleton_bones(skel_joints, skel_parents, radius=0.01)
    if len(skel_bone_verts) > 0:
        save_obj(skel_bone_verts, skel_bone_faces,
                 os.path.join(output_dir, 'skel_skeleton_bones.obj'))
        print(f"Saved: skel_skeleton_bones.obj")

    # =========================================================================
    # 3.5 Virtual Acromial (Introduced Virtual Joint)
    # =========================================================================
    print("\n=== Saving Virtual Acromial Joints ===")

    # Compute virtual acromial from SKEL mesh vertices
    skel_verts_tensor = torch.tensor(skel_verts, dtype=torch.float32).unsqueeze(0)  # [1, 6890, 3]
    virtual_r, virtual_l = compute_virtual_acromial(skel_verts_tensor)
    virtual_r = virtual_r[0].numpy()  # [3]
    virtual_l = virtual_l[0].numpy()  # [3]

    # Save virtual acromial as spheres (larger, different color conceptually)
    virtual_acromial_joints = np.stack([virtual_r, virtual_l], axis=0)  # [2, 3]
    virtual_acr_verts, virtual_acr_faces = create_joint_spheres(virtual_acromial_joints, radius=0.025)
    save_obj(virtual_acr_verts, virtual_acr_faces,
             os.path.join(output_dir, 'skel_virtual_acromial.obj'))
    print(f"Saved: skel_virtual_acromial.obj (SKEL virtual acromial from mesh vertices)")
    print(f"  Right: {virtual_r}")
    print(f"  Left:  {virtual_l}")

    # Also save the individual vertices used for virtual acromial computation
    v_idx_r = ACROMIAL_VERTEX_IDX['right']  # [4125, 4124, 5293, 5290]
    v_idx_l = ACROMIAL_VERTEX_IDX['left']   # [635, 636, 1830, 1829]
    acromial_verts_r = skel_verts[v_idx_r]  # [4, 3]
    acromial_verts_l = skel_verts[v_idx_l]  # [4, 3]
    all_acromial_verts = np.vstack([acromial_verts_r, acromial_verts_l])  # [8, 3]
    acr_vert_spheres, acr_vert_faces = create_joint_spheres(all_acromial_verts, radius=0.008)
    save_obj(acr_vert_spheres, acr_vert_faces,
             os.path.join(output_dir, 'skel_acromial_vertices.obj'))
    print(f"Saved: skel_acromial_vertices.obj (8 mesh vertices used for virtual acromial)")

    # Save AddB acromial joints for comparison
    # Find acromial indices in AddB
    addb_acr_r_idx = addb_names.index('acromial_r') if 'acromial_r' in addb_names else None
    addb_acr_l_idx = addb_names.index('acromial_l') if 'acromial_l' in addb_names else None

    if addb_acr_r_idx is not None and addb_acr_l_idx is not None:
        addb_acr_r = addb_joints_converted[addb_acr_r_idx]
        addb_acr_l = addb_joints_converted[addb_acr_l_idx]
        addb_acromial_joints = np.stack([addb_acr_r, addb_acr_l], axis=0)
        addb_acr_verts, addb_acr_faces = create_joint_spheres(addb_acromial_joints, radius=0.025)
        save_obj(addb_acr_verts, addb_acr_faces,
                 os.path.join(output_dir, 'addb_acromial.obj'))
        print(f"Saved: addb_acromial.obj (AddB explicit acromial joints)")
        print(f"  Right: {addb_acr_r}")
        print(f"  Left:  {addb_acr_l}")

        # Compute error
        err_r = np.linalg.norm(virtual_r - addb_acr_r) * 1000
        err_l = np.linalg.norm(virtual_l - addb_acr_l) * 1000
        print(f"\nVirtual Acromial Error:")
        print(f"  Right: {err_r:.2f} mm")
        print(f"  Left:  {err_l:.2f} mm")
        print(f"  Mean:  {(err_r + err_l) / 2:.2f} mm")

    # =========================================================================
    # 4. Print metrics
    # =========================================================================
    print("\n=== Metrics ===")
    for key, value in result['metrics'].items():
        print(f"  {key}: {value:.2f}")

    # Save joint positions as npy
    np.save(os.path.join(output_dir, 'addb_joints.npy'), addb_joints_converted)
    np.save(os.path.join(output_dir, 'skel_joints.npy'), skel_joints)
    np.save(os.path.join(output_dir, 'addb_names.npy'), np.array(addb_names))

    print(f"\n=== Done ===")
    print(f"Output directory: {output_dir}")
    print("Files:")
    print("  - addb_gt_joints.obj (AddB ground truth joints)")
    print("  - addb_gt_skeleton.obj (AddB ground truth skeleton)")
    print("  - addb_acromial.obj (AddB explicit acromial joints - TARGET)")
    print("  - skel_mesh.obj (SKEL skin mesh)")
    print("  - skel_skeleton_mesh.obj (SKEL internal skeleton)")
    print("  - skel_joints.obj (SKEL joint spheres)")
    print("  - skel_skeleton_bones.obj (SKEL bone connections)")
    print("  - skel_virtual_acromial.obj (SKEL virtual acromial - INTRODUCED JOINT)")
    print("  - skel_acromial_vertices.obj (8 mesh vertices used for virtual acromial)")


if __name__ == '__main__':
    main()
