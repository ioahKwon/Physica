#!/usr/bin/env python3
"""
Analyze Joint Correspondence between AddB and SKEL

Compare joint positions and bone directions to identify mapping issues.
"""

import os
import sys
import numpy as np
import torch
from typing import Dict, List, Tuple

sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/SKEL')
sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

from skel.skel_model import SKEL
from models.skel_model import AUTO_JOINT_NAME_MAP_SKEL, SKEL_JOINT_NAMES

# Lazy import nimblephysics to avoid early crash
nimble = None

# SKEL kinematic tree (parent indices)
SKEL_PARENTS = {
    0: -1,   # pelvis (root)
    1: 0,    # femur_r <- pelvis
    2: 1,    # tibia_r <- femur_r
    3: 2,    # talus_r <- tibia_r
    4: 3,    # calcn_r <- talus_r
    5: 4,    # toes_r <- calcn_r
    6: 0,    # femur_l <- pelvis
    7: 6,    # tibia_l <- femur_l
    8: 7,    # talus_l <- tibia_l
    9: 8,    # calcn_l <- talus_l
    10: 9,   # toes_l <- calcn_l
    11: 0,   # lumbar_body <- pelvis
    12: 11,  # thorax <- lumbar_body
    13: 12,  # head <- thorax
    14: 12,  # scapula_r <- thorax
    15: 14,  # humerus_r <- scapula_r
    16: 15,  # ulna_r <- humerus_r
    17: 16,  # radius_r <- ulna_r
    18: 17,  # hand_r <- radius_r
    19: 12,  # scapula_l <- thorax
    20: 19,  # humerus_l <- scapula_l
    21: 20,  # ulna_l <- humerus_l
    22: 21,  # radius_l <- ulna_l
    23: 22,  # hand_l <- radius_l
}


def load_addb_skeleton(b3d_path: str, frame: int = 0):
    """Load AddB skeleton from b3d file"""
    global nimble
    if nimble is None:
        import nimblephysics as nimble_mod
        nimble = nimble_mod

    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    skel = subject.readSkel(0)

    frames = subject.readFrames(trial=0, startFrame=frame, numFramesToRead=1, contactThreshold=20)
    pp = frames[0].processingPasses[0]
    pos = np.asarray(pp.pos, dtype=np.float32)
    skel.setPositions(pos)

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


def load_skel_skeleton(skel_params_path: str, skel_model_path: str, frame: int = 0):
    """Load SKEL skeleton from optimized params"""
    device = torch.device('cpu')

    skel = SKEL(model_path=skel_model_path, gender='male').to(device)

    params = np.load(skel_params_path)
    betas = torch.from_numpy(params['betas']).float().unsqueeze(0)
    poses = torch.from_numpy(params['poses'][frame:frame+1]).float()
    trans = torch.from_numpy(params['trans'][frame:frame+1]).float()

    output = skel(betas=betas, poses=poses, trans=trans)
    joints = output.joints[0].detach().cpu().numpy()

    return joints, SKEL_JOINT_NAMES, SKEL_PARENTS


def compute_bone_direction(joints: np.ndarray, parent_idx: int, child_idx: int) -> np.ndarray:
    """Compute normalized bone direction vector"""
    if parent_idx < 0:
        return np.zeros(3)
    direction = joints[child_idx] - joints[parent_idx]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.zeros(3)
    return direction / norm


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def save_combined_skeleton_obj(
    addb_joints: np.ndarray, addb_parents: List[int],
    skel_joints: np.ndarray, skel_parents: Dict[int, int],
    filepath: str
):
    """Save both skeletons in one OBJ file for comparison"""
    with open(filepath, 'w') as f:
        vertex_offset = 0

        # AddB skeleton (red spheres)
        f.write("# AddB Skeleton\n")
        for j_idx, joint in enumerate(addb_joints):
            # Small sphere
            phi = (1 + np.sqrt(5)) / 2
            r = 0.012
            sphere_verts = np.array([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ]) * r / np.sqrt(1 + phi**2) + joint

            for v in sphere_verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} 1.0 0.0 0.0\n")  # Red

            sphere_faces = [
                [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
            ]
            for face in sphere_faces:
                f.write(f"f {face[0]+1+vertex_offset} {face[1]+1+vertex_offset} {face[2]+1+vertex_offset}\n")
            vertex_offset += 12

        # AddB bones
        for i in range(len(addb_joints)):
            parent_idx = addb_parents[i]
            if parent_idx >= 0:
                p1, p2 = addb_joints[parent_idx], addb_joints[i]
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

                n_seg, bone_radius = 6, 0.006
                for end_point in [p1, p2]:
                    for seg in range(n_seg):
                        angle = 2 * np.pi * seg / n_seg
                        offset = bone_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                        v = end_point + offset
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} 1.0 0.3 0.3\n")  # Light red

                for seg in range(n_seg):
                    next_seg = (seg + 1) % n_seg
                    v1 = vertex_offset + seg
                    v2 = vertex_offset + next_seg
                    v3 = vertex_offset + n_seg + seg
                    v4 = vertex_offset + n_seg + next_seg
                    f.write(f"f {v1+1} {v2+1} {v3+1}\n")
                    f.write(f"f {v2+1} {v4+1} {v3+1}\n")
                vertex_offset += 2 * n_seg

        # SKEL skeleton (blue spheres)
        f.write("\n# SKEL Skeleton\n")
        for j_idx, joint in enumerate(skel_joints):
            phi = (1 + np.sqrt(5)) / 2
            r = 0.012
            sphere_verts = np.array([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ]) * r / np.sqrt(1 + phi**2) + joint

            for v in sphere_verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} 0.0 0.0 1.0\n")  # Blue

            sphere_faces = [
                [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
            ]
            for face in sphere_faces:
                f.write(f"f {face[0]+1+vertex_offset} {face[1]+1+vertex_offset} {face[2]+1+vertex_offset}\n")
            vertex_offset += 12

        # SKEL bones
        for i in range(len(skel_joints)):
            parent_idx = skel_parents.get(i, -1)
            if parent_idx >= 0:
                p1, p2 = skel_joints[parent_idx], skel_joints[i]
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

                n_seg, bone_radius = 6, 0.006
                for end_point in [p1, p2]:
                    for seg in range(n_seg):
                        angle = 2 * np.pi * seg / n_seg
                        offset = bone_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                        v = end_point + offset
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} 0.3 0.3 1.0\n")  # Light blue

                for seg in range(n_seg):
                    next_seg = (seg + 1) % n_seg
                    v1 = vertex_offset + seg
                    v2 = vertex_offset + next_seg
                    v3 = vertex_offset + n_seg + seg
                    v4 = vertex_offset + n_seg + next_seg
                    f.write(f"f {v1+1} {v2+1} {v3+1}\n")
                    f.write(f"f {v2+1} {v4+1} {v3+1}\n")
                vertex_offset += 2 * n_seg


def analyze_joint_correspondence(
    addb_joints: np.ndarray, addb_names: List[str], addb_parents: List[int],
    skel_joints: np.ndarray, skel_names: List[str], skel_parents: Dict[int, int],
    mapping: Dict[str, str]
):
    """Analyze joint correspondence between AddB and SKEL"""

    print("=" * 80)
    print("JOINT POSITION COMPARISON (AddB vs SKEL)")
    print("=" * 80)

    results = []

    # Build reverse mapping: SKEL name -> index
    skel_name_to_idx = {name: i for i, name in enumerate(skel_names)}
    addb_name_to_idx = {name: i for i, name in enumerate(addb_names)}

    for addb_name in addb_names:
        if addb_name not in mapping:
            continue

        skel_name = mapping[addb_name]
        if skel_name not in skel_name_to_idx:
            continue

        addb_idx = addb_name_to_idx[addb_name]
        skel_idx = skel_name_to_idx[skel_name]

        addb_pos = addb_joints[addb_idx]
        skel_pos = skel_joints[skel_idx]

        error_mm = np.linalg.norm(addb_pos - skel_pos) * 1000

        results.append({
            'addb_name': addb_name,
            'skel_name': skel_name,
            'addb_idx': addb_idx,
            'skel_idx': skel_idx,
            'addb_pos': addb_pos,
            'skel_pos': skel_pos,
            'error_mm': error_mm,
        })

    # Sort by error (descending)
    results.sort(key=lambda x: x['error_mm'], reverse=True)

    print(f"\n{'AddB Joint':<20} {'SKEL Joint':<15} {'Error (mm)':<12} {'AddB Pos':<30} {'SKEL Pos':<30}")
    print("-" * 110)

    for r in results:
        addb_pos_str = f"[{r['addb_pos'][0]:7.3f}, {r['addb_pos'][1]:7.3f}, {r['addb_pos'][2]:7.3f}]"
        skel_pos_str = f"[{r['skel_pos'][0]:7.3f}, {r['skel_pos'][1]:7.3f}, {r['skel_pos'][2]:7.3f}]"
        marker = "***" if r['error_mm'] > 50 else ""
        print(f"{r['addb_name']:<20} {r['skel_name']:<15} {r['error_mm']:>8.1f}    {addb_pos_str:<30} {skel_pos_str:<30} {marker}")

    # Bone direction analysis
    print("\n" + "=" * 80)
    print("BONE DIRECTION COMPARISON")
    print("=" * 80)

    # Define bone pairs to compare
    bone_pairs = [
        # (AddB parent, AddB child, SKEL parent, SKEL child)
        ('ground_pelvis', 'hip_r', 'pelvis', 'femur_r'),
        ('ground_pelvis', 'hip_l', 'pelvis', 'femur_l'),
        ('hip_r', 'walker_knee_r', 'femur_r', 'tibia_r'),
        ('hip_l', 'walker_knee_l', 'femur_l', 'tibia_l'),
        ('walker_knee_r', 'ankle_r', 'tibia_r', 'talus_r'),
        ('walker_knee_l', 'ankle_l', 'tibia_l', 'talus_l'),
        ('ground_pelvis', 'back', 'pelvis', 'lumbar_body'),
        ('back', 'acromial_r', 'thorax', 'humerus_r'),  # Key comparison
        ('back', 'acromial_l', 'thorax', 'humerus_l'),  # Key comparison
        ('acromial_r', 'elbow_r', 'humerus_r', 'ulna_r'),
        ('acromial_l', 'elbow_l', 'humerus_l', 'ulna_l'),
        ('elbow_r', 'radius_hand_r', 'ulna_r', 'hand_r'),
        ('elbow_l', 'radius_hand_l', 'ulna_l', 'hand_l'),
    ]

    print(f"\n{'AddB Bone':<30} {'SKEL Bone':<30} {'Cos Sim':<10} {'Angle (deg)':<12}")
    print("-" * 85)

    for addb_p, addb_c, skel_p, skel_c in bone_pairs:
        if addb_p not in addb_name_to_idx or addb_c not in addb_name_to_idx:
            continue
        if skel_p not in skel_name_to_idx or skel_c not in skel_name_to_idx:
            continue

        addb_dir = compute_bone_direction(
            addb_joints, addb_name_to_idx[addb_p], addb_name_to_idx[addb_c]
        )
        skel_dir = compute_bone_direction(
            skel_joints, skel_name_to_idx[skel_p], skel_name_to_idx[skel_c]
        )

        cos_sim = cosine_similarity(addb_dir, skel_dir)
        angle_deg = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))

        addb_bone = f"{addb_p} -> {addb_c}"
        skel_bone = f"{skel_p} -> {skel_c}"
        marker = "***" if angle_deg > 30 else ""

        print(f"{addb_bone:<30} {skel_bone:<30} {cos_sim:>8.3f}   {angle_deg:>8.1f}     {marker}")

    # Special analysis: Shoulder region
    print("\n" + "=" * 80)
    print("SHOULDER REGION ANALYSIS")
    print("=" * 80)

    # back vs thorax vs lumbar_body
    if 'back' in addb_name_to_idx:
        back_pos = addb_joints[addb_name_to_idx['back']]
        thorax_pos = skel_joints[skel_name_to_idx['thorax']]
        lumbar_pos = skel_joints[skel_name_to_idx['lumbar_body']]

        print(f"\nAddB 'back' position:      [{back_pos[0]:7.3f}, {back_pos[1]:7.3f}, {back_pos[2]:7.3f}]")
        print(f"SKEL 'thorax' position:    [{thorax_pos[0]:7.3f}, {thorax_pos[1]:7.3f}, {thorax_pos[2]:7.3f}]")
        print(f"SKEL 'lumbar_body' position: [{lumbar_pos[0]:7.3f}, {lumbar_pos[1]:7.3f}, {lumbar_pos[2]:7.3f}]")
        print(f"\nDistance back->thorax:     {np.linalg.norm(back_pos - thorax_pos)*1000:.1f} mm")
        print(f"Distance back->lumbar_body: {np.linalg.norm(back_pos - lumbar_pos)*1000:.1f} mm")

    # acromial vs humerus vs scapula
    for side in ['r', 'l']:
        acr_name = f'acromial_{side}'
        hum_name = f'humerus_{side}'
        scap_name = f'scapula_{side}'

        if acr_name in addb_name_to_idx:
            acr_pos = addb_joints[addb_name_to_idx[acr_name]]
            hum_pos = skel_joints[skel_name_to_idx[hum_name]]
            scap_pos = skel_joints[skel_name_to_idx[scap_name]]

            print(f"\n--- {side.upper()} Shoulder ---")
            print(f"AddB '{acr_name}':     [{acr_pos[0]:7.3f}, {acr_pos[1]:7.3f}, {acr_pos[2]:7.3f}]")
            print(f"SKEL '{hum_name}':     [{hum_pos[0]:7.3f}, {hum_pos[1]:7.3f}, {hum_pos[2]:7.3f}]")
            print(f"SKEL '{scap_name}':    [{scap_pos[0]:7.3f}, {scap_pos[1]:7.3f}, {scap_pos[2]:7.3f}]")
            print(f"Distance acromial->humerus: {np.linalg.norm(acr_pos - hum_pos)*1000:.1f} mm (current mapping)")
            print(f"Distance acromial->scapula: {np.linalg.norm(acr_pos - scap_pos)*1000:.1f} mm")

    return results


def main():
    # Paths
    b3d_path = '/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject1/Subject1.b3d'
    skel_params_path = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/skel/skel_params.npz'
    skel_model_path = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/joint_analysis'

    os.makedirs(out_dir, exist_ok=True)

    print("Loading AddB skeleton...")
    addb_joints, addb_names, addb_parents = load_addb_skeleton(b3d_path, frame=0)
    print(f"  Loaded {len(addb_joints)} joints")

    print("\nLoading SKEL skeleton...")
    skel_joints, skel_names, skel_parents = load_skel_skeleton(skel_params_path, skel_model_path, frame=0)
    print(f"  Loaded {len(skel_joints)} joints")

    # Print joint names
    print("\n" + "=" * 80)
    print("JOINT NAMES")
    print("=" * 80)
    print(f"\nAddB joints ({len(addb_names)}):")
    for i, name in enumerate(addb_names):
        parent = addb_names[addb_parents[i]] if addb_parents[i] >= 0 else "root"
        print(f"  {i:2d}: {name:<20} (parent: {parent})")

    print(f"\nSKEL joints ({len(skel_names)}):")
    for i, name in enumerate(skel_names):
        parent = skel_names[skel_parents[i]] if skel_parents.get(i, -1) >= 0 else "root"
        print(f"  {i:2d}: {name:<15} (parent: {parent})")

    # Analyze correspondence
    results = analyze_joint_correspondence(
        addb_joints, addb_names, addb_parents,
        skel_joints, skel_names, skel_parents,
        AUTO_JOINT_NAME_MAP_SKEL
    )

    # Save combined skeleton visualization
    vis_path = os.path.join(out_dir, 'skeleton_comparison.obj')
    save_combined_skeleton_obj(
        addb_joints, addb_parents,
        skel_joints, skel_parents,
        vis_path
    )
    print(f"\n\nVisualization saved to: {vis_path}")
    print("  Red = AddB, Blue = SKEL")


if __name__ == '__main__':
    main()
