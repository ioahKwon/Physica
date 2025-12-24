#!/usr/bin/env python3
"""
Compare different scapula regularization strategies:
1. elevation_only (current): Only regularize DOF 27, 37 (elevation)
2. no_scapula_reg: Remove scapula regularization entirely
3. all_scapula_dofs: Regularize all 6 scapula DOFs (26-28, 36-38)
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SKEL')

import nimblephysics as nimble

from models.skel_model import (
    SKELModelWrapper,
    SKEL_NUM_BETAS,
    SKEL_NUM_POSE_DOF,
    SKEL_JOINT_NAMES,
    AUTO_JOINT_NAME_MAP_SKEL,
)

from compare_smpl_skel import (
    build_joint_mapping,
    build_bone_pair_mapping,
    build_bone_length_pairs,
    compute_bone_direction_loss,
    compute_bone_length_loss,
    compute_mpjpe,
    SKEL_BONE_PAIRS,
    SKEL_ADDB_BONE_LENGTH_PAIRS,
    SKEL_MODEL_PATH,
)

from shoulder_correction import (
    ACROMIAL_VERTEX_IDX,
    compute_virtual_acromial,
    loss_acromial,
    loss_shoulder_width_from_vertices,
)


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


def load_addb_frame(b3d_path, trial_idx=0, frame_idx=0):
    """Load single frame from AddB b3d file"""
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    print(f"Number of trials: {subject.getNumTrials()}")
    print(f"Loading trial {trial_idx}, frame {frame_idx}")

    trial_length = subject.getTrialLength(trial_idx)
    if frame_idx >= trial_length:
        print(f"Warning: frame_idx {frame_idx} >= trial_length {trial_length}, using last frame")
        frame_idx = trial_length - 1

    frames = subject.readFrames(trial_idx, frame_idx, 1)
    frame = frames[0]
    skel = subject.readSkel(0)

    pos = frame.processingPasses[0].pos
    skel.setPositions(pos)

    joint_names = []
    joint_positions = []
    for i in range(skel.getNumJoints()):
        joint = skel.getJoint(i)
        joint_names.append(joint.getName())
        child_body = joint.getChildBodyNode()
        world_pos = child_body.getWorldTransform().translation()
        joint_positions.append(world_pos)

    return np.array(joint_positions), joint_names


def optimize_with_scapula_reg(
    target_joints,
    addb_joint_names,
    device,
    scapula_reg_mode='elevation_only',  # 'elevation_only', 'no_scapula_reg', 'all_scapula_dofs'
    num_iters=300,
    verbose=True,
):
    """
    Optimize SKEL with different scapula regularization modes.
    """
    if verbose:
        print(f"\n=== Optimization with scapula_reg_mode='{scapula_reg_mode}' ===")

    skel = SKELModelWrapper(model_path=SKEL_MODEL_PATH, gender='male', device=device)

    addb_indices_full, skel_indices_full = build_joint_mapping(
        addb_joint_names, SKEL_JOINT_NAMES, AUTO_JOINT_NAME_MAP_SKEL
    )

    acr_r_idx = addb_joint_names.index('acromial_r') if 'acromial_r' in addb_joint_names else None
    acr_l_idx = addb_joint_names.index('acromial_l') if 'acromial_l' in addb_joint_names else None
    has_acromial = acr_r_idx is not None and acr_l_idx is not None

    addb_indices = addb_indices_full
    skel_indices = skel_indices_full

    # Exclude acromial from joint loss (use virtual acromial loss instead)
    addb_indices_for_loss = []
    skel_indices_for_loss = []
    for ai, si in zip(addb_indices_full, skel_indices_full):
        if ai not in [acr_r_idx, acr_l_idx]:
            addb_indices_for_loss.append(ai)
            skel_indices_for_loss.append(si)

    bone_pairs = build_bone_pair_mapping(
        addb_joint_names, SKEL_JOINT_NAMES, AUTO_JOINT_NAME_MAP_SKEL, SKEL_BONE_PAIRS
    )

    bone_length_pairs = build_bone_length_pairs(
        addb_joint_names, SKEL_JOINT_NAMES, SKEL_ADDB_BONE_LENGTH_PAIRS
    )

    # Initialize parameters
    T = len(target_joints)
    target = torch.tensor(target_joints, dtype=torch.float32, device=device)

    betas = torch.zeros(1, SKEL_NUM_BETAS, device=device, requires_grad=True)
    poses = torch.zeros(T, SKEL_NUM_POSE_DOF, device=device, requires_grad=True)
    trans = torch.zeros(T, 3, device=device)

    # Initialize trans to pelvis location
    pelvis_idx = addb_joint_names.index('ground_pelvis') if 'ground_pelvis' in addb_joint_names else 0
    with torch.no_grad():
        trans[:] = target[:, pelvis_idx]
    trans.requires_grad = True

    optimizer = torch.optim.Adam([betas, poses, trans], lr=0.01)

    # Acromial targets
    if has_acromial:
        acr_target_r = target[:, acr_r_idx, :]
        acr_target_l = target[:, acr_l_idx, :]

    for it in range(num_iters):
        optimizer.zero_grad()

        verts, pred_joints = skel.forward(betas.expand(T, -1), poses, trans)

        # Debug: print SKEL output stats on first iteration
        if it == 0 and verbose:
            with torch.no_grad():
                pj = pred_joints.cpu().numpy()
                print(f"  SKEL joints: X [{pj[0, :, 0].min():.3f}, {pj[0, :, 0].max():.3f}], "
                      f"Y [{pj[0, :, 1].min():.3f}, {pj[0, :, 1].max():.3f}], "
                      f"Z [{pj[0, :, 2].min():.3f}, {pj[0, :, 2].max():.3f}]")
                tj = target.cpu().numpy()
                print(f"  Target joints: X [{tj[0, :, 0].min():.3f}, {tj[0, :, 0].max():.3f}], "
                      f"Y [{tj[0, :, 1].min():.3f}, {tj[0, :, 1].max():.3f}], "
                      f"Z [{tj[0, :, 2].min():.3f}, {tj[0, :, 2].max():.3f}]")

        # Joint loss (excluding acromial)
        joint_loss = ((pred_joints[:, skel_indices_for_loss, :] -
                       target[:, addb_indices_for_loss, :]) ** 2).mean()

        # Bone losses
        bone_dir_loss = compute_bone_direction_loss(
            pred_joints, target, bone_pairs
        )
        bone_len_loss = compute_bone_length_loss(
            pred_joints, target, bone_length_pairs
        )

        # Shoulder loss
        if has_acromial:
            shoulder_loss = loss_acromial(verts, acr_target_r, acr_target_l)
            width_loss = loss_shoulder_width_from_vertices(
                verts, acr_target_r, acr_target_l
            )
        else:
            shoulder_loss = torch.tensor(0.0, device=device)
            width_loss = torch.tensor(0.0, device=device)

        # Total loss
        loss = joint_loss + 0.5 * bone_dir_loss + 0.3 * bone_len_loss
        loss = loss + 1.0 * shoulder_loss + 0.5 * width_loss

        # Standard regularization
        loss = loss + 0.01 * (poses ** 2).mean()
        loss = loss + 0.005 * (betas ** 2).mean()

        # Spine regularization
        spine_dof_indices = list(range(17, 26))
        spine_dofs = poses[:, spine_dof_indices]
        loss = loss + 0.05 * (spine_dofs ** 2).mean()

        # Scapula regularization based on mode
        if scapula_reg_mode == 'elevation_only':
            # Current: only elevation (DOF 27, 37)
            scapula_indices = [27, 37]
            scapula_dofs = poses[:, scapula_indices]
            loss = loss + 0.1 * (scapula_dofs ** 2).mean()
        elif scapula_reg_mode == 'all_scapula_dofs':
            # All 6 scapula DOFs: abduction(26,36), elevation(27,37), upward_rot(28,38)
            scapula_indices = [26, 27, 28, 36, 37, 38]
            scapula_dofs = poses[:, scapula_indices]
            loss = loss + 0.1 * (scapula_dofs ** 2).mean()
        elif scapula_reg_mode == 'no_scapula_reg':
            # No scapula regularization
            pass

        loss.backward()
        optimizer.step()

        if verbose and ((it + 1) % 50 == 0 or it == 0):
            with torch.no_grad():
                mpjpe = compute_mpjpe(
                    pred_joints.detach().cpu().numpy(),
                    target_joints, skel_indices, addb_indices
                )
            print(f"  Iter {it+1}/{num_iters}: Loss={loss.item():.4f}, MPJPE={mpjpe:.1f}mm")

    # Final results
    with torch.no_grad():
        verts, pred_joints = skel.forward(betas.expand(T, -1), poses, trans)
        verts_np = verts.cpu().numpy()
        joints_np = pred_joints.cpu().numpy()
        poses_np = poses.cpu().numpy()
        betas_np = betas.cpu().numpy()

        mpjpe = compute_mpjpe(joints_np, target_joints, skel_indices, addb_indices)

        if has_acromial:
            virtual_r, virtual_l = compute_virtual_acromial(verts)
            vr = virtual_r.cpu().numpy()
            vl = virtual_l.cpu().numpy()
            err_r = np.linalg.norm(vr - acr_target_r.cpu().numpy(), axis=-1).mean() * 1000
            err_l = np.linalg.norm(vl - acr_target_l.cpu().numpy(), axis=-1).mean() * 1000
            virtual_err = (err_r + err_l) / 2
        else:
            virtual_err = 0.0

    return {
        'vertices': verts_np,
        'joints': joints_np,
        'poses': poses_np,
        'betas': betas_np,
        'faces': skel.faces,
        'skel_faces': skel.skel_faces,
        'parents': skel.parents.cpu().numpy().tolist(),
        'mpjpe_mm': mpjpe,  # Already in mm from compute_mpjpe
        'virtual_acromial_err_mm': virtual_err,
        'scapula_dofs': {
            'abduction_r': poses_np[0, 26],
            'elevation_r': poses_np[0, 27],
            'upward_rot_r': poses_np[0, 28],
            'abduction_l': poses_np[0, 36],
            'elevation_l': poses_np[0, 37],
            'upward_rot_l': poses_np[0, 38],
        }
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default='Subject1')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--output_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/03_Output/scapula_reg_comparison')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load AddB data
    b3d_path = f'/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/{args.subject}/{args.subject}.b3d'
    addb_joints, addb_names = load_addb_frame(b3d_path, args.trial, args.frame)

    print(f"\n=== AddB Joint Stats (raw) ===")
    print(f"Shape: {addb_joints.shape}")
    print(f"X range: [{addb_joints[:, 0].min():.3f}, {addb_joints[:, 0].max():.3f}]")
    print(f"Y range: [{addb_joints[:, 1].min():.3f}, {addb_joints[:, 1].max():.3f}]")
    print(f"Z range: [{addb_joints[:, 2].min():.3f}, {addb_joints[:, 2].max():.3f}]")

    # Coordinate conversion: AddB -> SKEL
    addb_joints_converted = np.zeros_like(addb_joints)
    addb_joints_converted[:, 0] = -addb_joints[:, 2]
    addb_joints_converted[:, 1] = addb_joints[:, 1]
    addb_joints_converted[:, 2] = addb_joints[:, 0]
    addb_joints_converted = addb_joints_converted[np.newaxis, :, :]  # [1, N, 3]

    print(f"\n=== AddB Joint Stats (converted to SKEL coords) ===")
    print(f"Shape: {addb_joints_converted.shape}")
    print(f"X range: [{addb_joints_converted[0, :, 0].min():.3f}, {addb_joints_converted[0, :, 0].max():.3f}]")
    print(f"Y range: [{addb_joints_converted[0, :, 1].min():.3f}, {addb_joints_converted[0, :, 1].max():.3f}]")
    print(f"Z range: [{addb_joints_converted[0, :, 2].min():.3f}, {addb_joints_converted[0, :, 2].max():.3f}]")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f'{args.subject}_frame{args.frame}')
    os.makedirs(output_dir, exist_ok=True)

    # Run experiments
    experiments = [
        ('elevation_only', 'elevation_only'),
        ('no_scapula_reg', 'no_scapula_reg'),
        ('all_scapula_dofs', 'all_scapula_dofs'),
    ]
    results = {}

    for mode_name, scapula_mode in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {mode_name}")
        print('='*60)

        result = optimize_with_scapula_reg(
            addb_joints_converted,
            addb_names,
            device,
            scapula_reg_mode=scapula_mode,
            num_iters=args.iters,
            verbose=True,
        )
        results[mode_name] = result

        # Save outputs
        mode_dir = os.path.join(output_dir, mode_name)
        os.makedirs(mode_dir, exist_ok=True)

        # Save mesh
        save_obj(result['vertices'][0], result['faces'],
                 os.path.join(mode_dir, 'mesh.obj'))

        # Save joints
        joint_verts, joint_faces = create_joint_spheres(result['joints'][0], radius=0.015)
        save_obj(joint_verts, joint_faces, os.path.join(mode_dir, 'joints.obj'))

        # Save skeleton
        bone_verts, bone_faces = create_skeleton_bones(
            result['joints'][0], result['parents'], radius=0.008
        )
        if len(bone_verts) > 0:
            save_obj(bone_verts, bone_faces, os.path.join(mode_dir, 'skeleton.obj'))

        # Save skeleton mesh
        if result['skel_faces'] is not None:
            # Get skeleton mesh
            skel_model = SKELModelWrapper(model_path=SKEL_MODEL_PATH, gender='male', device=device)
            poses_t = torch.tensor(result['poses'], dtype=torch.float32, device=device)
            betas_t = torch.tensor(result['betas'], dtype=torch.float32, device=device)
            trans_t = torch.zeros(1, 3, device=device)
            _, _, skel_verts = skel_model.forward(betas_t, poses_t, trans_t, return_skeleton=True)
            save_obj(skel_verts[0].cpu().numpy(), result['skel_faces'],
                     os.path.join(mode_dir, 'skeleton_mesh.obj'))

        print(f"  Saved to: {mode_dir}/")

    # Print comparison summary
    mode_names = [e[0] for e in experiments]
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Mode':<25} {'MPJPE (mm)':<15} {'Virtual Acr Err (mm)':<20}")
    print("-"*60)
    for mode in mode_names:
        r = results[mode]
        print(f"{mode:<25} {r['mpjpe_mm']:<15.2f} {r['virtual_acromial_err_mm']:<20.2f}")

    print("\n" + "="*80)
    print("SCAPULA DOF VALUES (radians)")
    print("="*80)
    print(f"{'Mode':<25} {'abd_R':<10} {'elev_R':<10} {'uprot_R':<10} {'abd_L':<10} {'elev_L':<10} {'uprot_L':<10}")
    print("-"*85)
    for mode in mode_names:
        d = results[mode]['scapula_dofs']
        print(f"{mode:<25} {d['abduction_r']:<10.3f} {d['elevation_r']:<10.3f} {d['upward_rot_r']:<10.3f} "
              f"{d['abduction_l']:<10.3f} {d['elevation_l']:<10.3f} {d['upward_rot_l']:<10.3f}")

    # Save AddB GT for comparison
    addb_joint_verts, addb_joint_faces = create_joint_spheres(addb_joints_converted[0], radius=0.02)
    save_obj(addb_joint_verts, addb_joint_faces, os.path.join(output_dir, 'addb_gt_joints.obj'))

    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
