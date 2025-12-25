#!/usr/bin/env python
"""
Run the best AddB to SKEL pipeline (MPJPE ~21mm).

This uses the compare_smpl_skel.py algorithm with optimal settings.
"""

import os
import sys
import argparse

# Add paths
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

def main():
    parser = argparse.ArgumentParser(description='Run best AddB to SKEL pipeline')
    parser.add_argument('--b3d', type=str, required=True, help='Path to .b3d file')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames')
    parser.add_argument('--num_iters', type=int, default=400, help='Optimization iterations')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save_every', type=int, default=1, help='Save every N frames')
    parser.add_argument('--skel_only', action='store_true', help='Skip SMPL optimization')
    args = parser.parse_args()

    # Import here to avoid loading heavy modules at startup
    from compare_smpl_skel import (
        load_b3d_data, extract_body_proportions, optimize_skel,
        create_joint_spheres, create_skeleton_bones, save_obj
    )
    import numpy as np
    import torch

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    print("\n=== Loading AddB Data ===")
    target_joints, joint_names, addb_parents, subject_info = load_b3d_data(args.b3d, args.num_frames)

    # Extract body proportions
    body_proportions = extract_body_proportions(target_joints, joint_names)
    print(f"\nBody proportions:")
    for k, v in body_proportions.items():
        print(f"  {k}: {v*1000:.1f} mm")

    # Get gender
    sex = subject_info.get('sex', 'male')
    gender = sex if sex in ['male', 'female'] else 'male'
    print(f"\nSubject: height={subject_info.get('height_m', 'N/A')}m, "
          f"mass={subject_info.get('mass_kg', 'N/A')}kg, sex={sex}")

    # Run SKEL optimization with best settings
    print("\n=== Running SKEL Optimization ===")
    skel_result = optimize_skel(
        target_joints, joint_names, device, args.num_iters,
        virtual_acromial_weight=0.0,       # acromial→humerus direct mapping
        shoulder_width_weight=10.0,        # Force shoulder width to match AddB
        use_beta_init=True,
        use_dynamic_virtual_acromial=False,
        gender=gender,
        subject_info=subject_info,
        body_proportions=body_proportions
    )

    # Save results
    print("\n=== Saving Results ===")

    # Save parameters
    np.savez(
        os.path.join(args.out_dir, 'skel_params.npz'),
        betas=skel_result['betas'],
        poses=skel_result['poses'],
        trans=skel_result['trans'],
        joints=skel_result['joints']
    )

    # Save meshes
    num_saved = 0
    for t in range(0, len(skel_result['vertices']), args.save_every):
        # Skin mesh
        if skel_result['faces'] is not None:
            save_obj(
                skel_result['vertices'][t],
                skel_result['faces'],
                os.path.join(args.out_dir, f'skel_skin_frame{t:03d}.obj')
            )

        # Skeleton mesh (bones)
        if 'skel_vertices' in skel_result and skel_result['skel_faces'] is not None:
            save_obj(
                skel_result['skel_vertices'][t],
                skel_result['skel_faces'],
                os.path.join(args.out_dir, f'skel_bones_frame{t:03d}.obj')
            )

        # Joint skeleton visualization
        if 'parents' in skel_result:
            joints_t = skel_result['joints'][t]
            verts, faces = create_skeleton_bones(joints_t, skel_result['parents'], radius=0.01)
            if len(verts) > 0:
                save_obj(verts, faces, os.path.join(args.out_dir, f'skel_skeleton_frame{t:03d}.obj'))

        num_saved += 1

    print(f"  Saved {num_saved} frames to {args.out_dir}")

    # Save AddB skeleton for comparison
    addb_dir = os.path.join(args.out_dir, 'addb')
    os.makedirs(addb_dir, exist_ok=True)
    for t in range(0, len(target_joints), args.save_every):
        verts, faces = create_skeleton_bones(target_joints[t], addb_parents, radius=0.01)
        if len(verts) > 0:
            save_obj(verts, faces, os.path.join(addb_dir, f'skeleton_frame{t:03d}.obj'))

    print(f"\nDone! Results saved to {args.out_dir}")


if __name__ == '__main__':
    main()
