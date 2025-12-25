"""
Analyze bone length differences between AddB and SKEL.

Compare arm bone lengths to understand scaling issues.
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add paths
SKEL_REPO_PATH = '/egr/research-zijunlab/kwonjoon/01_Code/SKEL'
if SKEL_REPO_PATH not in sys.path:
    sys.path.insert(0, SKEL_REPO_PATH)

ADDB2SKEL_PATH = '/egr/research-zijunlab/kwonjoon/01_Code/Physica/addb2skel'
if ADDB2SKEL_PATH not in sys.path:
    sys.path.insert(0, ADDB2SKEL_PATH)

from skel.skel_model import SKEL
from utils.geometry import convert_addb_to_skel_coords
from utils.io import load_b3d
from config import SKEL_MODEL_PATH


def compute_bone_length(joints, idx1, idx2):
    """Compute bone length between two joints."""
    return np.linalg.norm(joints[idx1] - joints[idx2])


def main():
    parser = argparse.ArgumentParser(description='Analyze bone lengths')
    parser.add_argument('--params_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/best_scapula_mapping_21mm/skel_params.npz',
                        help='Path to optimized SKEL parameters')
    parser.add_argument('--b3d_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject11/Subject11.b3d',
                        help='Path to AddB b3d file')
    parser.add_argument('--gender', type=str, default='male', help='Gender for SKEL model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load SKEL model
    print("Loading SKEL model...")
    skel_model = SKEL(
        gender=args.gender,
        model_path=SKEL_MODEL_PATH,
    ).to(device)
    skel_model.eval()

    # Load optimized parameters
    print(f"Loading SKEL params from {args.params_path}...")
    params = np.load(args.params_path)
    poses = torch.from_numpy(params['poses']).float().to(device)
    betas = torch.from_numpy(params['betas']).float().to(device)
    trans = torch.from_numpy(params['trans']).float().to(device)

    # Load AddB joints
    print(f"Loading AddB joints from {args.b3d_path}...")
    addb_joints_raw, joint_names, metadata = load_b3d(args.b3d_path)
    addb_joints = convert_addb_to_skel_coords(addb_joints_raw)

    print(f"\nAddB joint names: {joint_names}")

    # AddB joint indices (from the joint names)
    # Typical: pelvis, hip_r, walker_knee_r, ankle_r, subtalar_r, mtp_r,
    #          hip_l, walker_knee_l, ankle_l, subtalar_l, mtp_l,
    #          back, acromial_r, elbow_r, radioulnar_r, radius_hand_r,
    #          acromial_l, elbow_l, radioulnar_l, radius_hand_l
    ADDB_ACROMIAL_R = 12
    ADDB_ELBOW_R = 13
    ADDB_RADIOULNAR_R = 14
    ADDB_HAND_R = 15
    ADDB_ACROMIAL_L = 16
    ADDB_ELBOW_L = 17
    ADDB_RADIOULNAR_L = 18
    ADDB_HAND_L = 19

    # SKEL joint indices
    SKEL_SCAPULA_R = 14
    SKEL_HUMERUS_R = 15
    SKEL_ULNA_R = 16
    SKEL_RADIUS_R = 17
    SKEL_HAND_R = 18
    SKEL_SCAPULA_L = 19
    SKEL_HUMERUS_L = 20
    SKEL_ULNA_L = 21
    SKEL_RADIUS_L = 22
    SKEL_HAND_L = 23

    # Get SKEL joints
    with torch.no_grad():
        output = skel_model(
            poses=poses[0:1],
            betas=betas.unsqueeze(0),
            trans=trans[0:1],
            poses_type='skel',
            skelmesh=False,
        )
    skel_joints = output.joints[0].cpu().numpy()

    # Get AddB joints for first frame
    addb_frame = addb_joints[0]

    print("\n" + "="*60)
    print("BONE LENGTH COMPARISON (Frame 0)")
    print("="*60)

    # Right arm
    print("\n--- RIGHT ARM ---")

    # AddB arm bones
    addb_upper_arm_r = compute_bone_length(addb_frame, ADDB_ACROMIAL_R, ADDB_ELBOW_R)
    addb_forearm_r = compute_bone_length(addb_frame, ADDB_ELBOW_R, ADDB_RADIOULNAR_R)
    addb_wrist_r = compute_bone_length(addb_frame, ADDB_RADIOULNAR_R, ADDB_HAND_R)
    addb_total_arm_r = addb_upper_arm_r + addb_forearm_r + addb_wrist_r

    print(f"AddB Upper arm (acromial→elbow):    {addb_upper_arm_r*1000:.1f} mm")
    print(f"AddB Forearm (elbow→radioulnar):    {addb_forearm_r*1000:.1f} mm")
    print(f"AddB Wrist (radioulnar→hand):       {addb_wrist_r*1000:.1f} mm")
    print(f"AddB Total arm:                     {addb_total_arm_r*1000:.1f} mm")

    # SKEL arm bones (using scapula as shoulder)
    skel_scap_to_hum_r = compute_bone_length(skel_joints, SKEL_SCAPULA_R, SKEL_HUMERUS_R)
    skel_upper_arm_r = compute_bone_length(skel_joints, SKEL_HUMERUS_R, SKEL_ULNA_R)
    skel_forearm_r = compute_bone_length(skel_joints, SKEL_ULNA_R, SKEL_RADIUS_R)
    skel_wrist_r = compute_bone_length(skel_joints, SKEL_RADIUS_R, SKEL_HAND_R)
    skel_total_arm_r = skel_scap_to_hum_r + skel_upper_arm_r + skel_forearm_r + skel_wrist_r

    print(f"\nSKEL Scapula→Humerus:               {skel_scap_to_hum_r*1000:.1f} mm")
    print(f"SKEL Upper arm (humerus→ulna):      {skel_upper_arm_r*1000:.1f} mm")
    print(f"SKEL Forearm (ulna→radius):         {skel_forearm_r*1000:.1f} mm")
    print(f"SKEL Wrist (radius→hand):           {skel_wrist_r*1000:.1f} mm")
    print(f"SKEL Total arm:                     {skel_total_arm_r*1000:.1f} mm")

    # Comparison: AddB acromial→elbow vs SKEL scapula→ulna
    skel_shoulder_to_elbow = skel_scap_to_hum_r + skel_upper_arm_r
    print(f"\n[Comparison]")
    print(f"AddB acromial→elbow:                {addb_upper_arm_r*1000:.1f} mm")
    print(f"SKEL scapula→ulna:                  {skel_shoulder_to_elbow*1000:.1f} mm")
    print(f"  Difference:                       {(addb_upper_arm_r - skel_shoulder_to_elbow)*1000:.1f} mm ({(addb_upper_arm_r/skel_shoulder_to_elbow - 1)*100:.1f}%)")

    # AddB elbow→hand vs SKEL ulna→hand
    addb_elbow_to_hand = addb_forearm_r + addb_wrist_r
    skel_elbow_to_hand = skel_forearm_r + skel_wrist_r
    print(f"\nAddB elbow→hand:                    {addb_elbow_to_hand*1000:.1f} mm")
    print(f"SKEL ulna→hand:                     {skel_elbow_to_hand*1000:.1f} mm")
    print(f"  Difference:                       {(addb_elbow_to_hand - skel_elbow_to_hand)*1000:.1f} mm ({(addb_elbow_to_hand/skel_elbow_to_hand - 1)*100:.1f}%)")

    # Left arm
    print("\n--- LEFT ARM ---")

    addb_upper_arm_l = compute_bone_length(addb_frame, ADDB_ACROMIAL_L, ADDB_ELBOW_L)
    addb_forearm_l = compute_bone_length(addb_frame, ADDB_ELBOW_L, ADDB_RADIOULNAR_L)
    addb_wrist_l = compute_bone_length(addb_frame, ADDB_RADIOULNAR_L, ADDB_HAND_L)

    skel_scap_to_hum_l = compute_bone_length(skel_joints, SKEL_SCAPULA_L, SKEL_HUMERUS_L)
    skel_upper_arm_l = compute_bone_length(skel_joints, SKEL_HUMERUS_L, SKEL_ULNA_L)
    skel_forearm_l = compute_bone_length(skel_joints, SKEL_ULNA_L, SKEL_RADIUS_L)
    skel_wrist_l = compute_bone_length(skel_joints, SKEL_RADIUS_L, SKEL_HAND_L)

    print(f"AddB Upper arm (acromial→elbow):    {addb_upper_arm_l*1000:.1f} mm")
    print(f"SKEL Scapula→Ulna:                  {(skel_scap_to_hum_l + skel_upper_arm_l)*1000:.1f} mm")
    print(f"  Difference:                       {(addb_upper_arm_l - skel_scap_to_hum_l - skel_upper_arm_l)*1000:.1f} mm")

    print(f"\nAddB elbow→hand:                    {(addb_forearm_l + addb_wrist_l)*1000:.1f} mm")
    print(f"SKEL ulna→hand:                     {(skel_forearm_l + skel_wrist_l)*1000:.1f} mm")
    print(f"  Difference:                       {(addb_forearm_l + addb_wrist_l - skel_forearm_l - skel_wrist_l)*1000:.1f} mm")

    # Overall arm scale factor
    print("\n--- SCALE FACTORS ---")
    scale_upper_r = addb_upper_arm_r / skel_shoulder_to_elbow
    scale_lower_r = addb_elbow_to_hand / skel_elbow_to_hand
    print(f"Upper arm scale (R): {scale_upper_r:.3f}")
    print(f"Lower arm scale (R): {scale_lower_r:.3f}")

    # Per-joint position errors
    print("\n--- JOINT POSITION ERRORS (R) ---")

    # Compare mapped joints
    mappings = [
        ("acromial→scapula", ADDB_ACROMIAL_R, SKEL_SCAPULA_R),
        ("elbow→ulna", ADDB_ELBOW_R, SKEL_ULNA_R),
        ("radioulnar→radius", ADDB_RADIOULNAR_R, SKEL_RADIUS_R),
        ("hand→hand", ADDB_HAND_R, SKEL_HAND_R),
    ]

    for name, addb_idx, skel_idx in mappings:
        error = np.linalg.norm(addb_frame[addb_idx] - skel_joints[skel_idx])
        print(f"  {name}: {error*1000:.1f} mm")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
