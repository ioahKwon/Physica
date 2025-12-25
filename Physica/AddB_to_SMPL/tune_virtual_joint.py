#!/usr/bin/env python3
"""
Parameter tuning for Virtual Joint optimization
"""

import os
import sys
import numpy as np
import torch
import json

sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

from compare_smpl_skel import (
    load_b3d_data, extract_body_proportions, optimize_skel, compute_mpjpe
)

def run_experiment(b3d_path: str, out_dir: str,
                   virtual_joint_weight: float,
                   shoulder_width_weight: float,
                   num_iters: int = 200,
                   num_frames: int = 5):
    """Run a single experiment with given parameters"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    target_joints, joint_names, addb_parents, subject_info = load_b3d_data(b3d_path, num_frames)
    body_proportions = extract_body_proportions(target_joints, joint_names)

    sex = subject_info.get('sex', 'male')
    gender = sex if sex in ['male', 'female'] else 'male'

    # Run optimization
    result = optimize_skel(
        target_joints, joint_names, device, num_iters,
        virtual_acromial_weight=0.0,
        shoulder_width_weight=shoulder_width_weight,
        use_beta_init=True,
        use_dynamic_virtual_acromial=False,
        gender=gender,
        subject_info=subject_info,
        body_proportions=body_proportions,
        use_virtual_joints=True,
        virtual_joint_weight=virtual_joint_weight
    )

    # Compute MPJPE
    mpjpe = compute_mpjpe(result['joints'], target_joints,
                          result['model_indices'], result['addb_indices'])

    return {
        'mpjpe': mpjpe,
        'alpha_shoulder': result.get('alpha_shoulder'),
        'alpha_wrist': result.get('alpha_wrist'),
        'betas_norm': np.linalg.norm(result['betas'])
    }


def main():
    b3d_path = '/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject1/Subject1.b3d'
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/tuning_results'
    os.makedirs(out_dir, exist_ok=True)

    # Parameter grid
    virtual_joint_weights = [1.0, 5.0, 10.0, 20.0, 50.0]
    shoulder_width_weights = [5.0, 10.0, 20.0]

    results = []

    print("=" * 80)
    print("Virtual Joint Parameter Tuning")
    print("=" * 80)

    for vj_w in virtual_joint_weights:
        for sw_w in shoulder_width_weights:
            print(f"\n>>> Testing: virtual_joint_weight={vj_w}, shoulder_width_weight={sw_w}")

            try:
                result = run_experiment(
                    b3d_path, out_dir,
                    virtual_joint_weight=vj_w,
                    shoulder_width_weight=sw_w,
                    num_iters=200,
                    num_frames=5
                )

                results.append({
                    'virtual_joint_weight': vj_w,
                    'shoulder_width_weight': sw_w,
                    **result
                })

                print(f"    MPJPE: {result['mpjpe']:.1f} mm")
                print(f"    α_shoulder: {result['alpha_shoulder']:.3f}")
                print(f"    α_wrist: {result['alpha_wrist']:.3f}")

            except Exception as e:
                print(f"    Error: {e}")

    # Sort by MPJPE
    results.sort(key=lambda x: x['mpjpe'])

    print("\n" + "=" * 80)
    print("Results Summary (sorted by MPJPE)")
    print("=" * 80)
    print(f"{'VJ Weight':>10} {'SW Weight':>10} {'MPJPE':>10} {'α_sh':>8} {'α_wr':>8}")
    print("-" * 50)

    for r in results:
        print(f"{r['virtual_joint_weight']:>10.1f} {r['shoulder_width_weight']:>10.1f} "
              f"{r['mpjpe']:>10.1f} {r['alpha_shoulder']:>8.3f} {r['alpha_wrist']:>8.3f}")

    # Save results
    with open(os.path.join(out_dir, 'tuning_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBest result: MPJPE={results[0]['mpjpe']:.1f}mm with "
          f"vj_weight={results[0]['virtual_joint_weight']}, sw_weight={results[0]['shoulder_width_weight']}")


if __name__ == '__main__':
    main()
