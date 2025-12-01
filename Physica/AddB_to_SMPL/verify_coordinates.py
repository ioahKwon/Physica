#!/usr/bin/env python3
"""
Verify coordinate system conversion by comparing AddB and SMPL joint positions
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel
import nimblephysics as nimble


def read_addb_joints(b3d_path, trial_idx=0, frame_idx=0):
    """Read AddBiomechanics joint positions for a specific frame"""
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    trial_length = subject.getTrialLength(trial_idx)

    frames = subject.readFrames(
        trial=trial_idx,
        startFrame=frame_idx,
        numFramesToRead=1,
        stride=1,
        includeSensorData=False,
        includeProcessingPasses=True
    )

    if len(frames) > 0 and len(frames[0].processingPasses) > 0:
        jc = np.asarray(frames[0].processingPasses[0].jointCenters, dtype=np.float32)
        joint_positions = jc.reshape(-1, 3)
        return joint_positions
    return None


def read_smpl_joints(smpl_params_path, smpl_model_path, frame_idx=0):
    """Read SMPL joint positions for a specific frame"""
    device = torch.device('cpu')
    smpl_model = SMPLModel(smpl_model_path, device=device)

    data = np.load(smpl_params_path)
    poses = data['poses']
    trans = data['trans']

    # Extract frame
    betas_frame = np.zeros((1, 10), dtype=np.float32)

    if len(poses.shape) == 3:
        poses_frame = poses[frame_idx:frame_idx+1].reshape(1, -1)
    else:
        poses_frame = poses[frame_idx:frame_idx+1]

    trans_frame = trans[frame_idx:frame_idx+1]

    # Convert to tensors
    betas_t = torch.from_numpy(betas_frame).float().to(device)
    poses_t = torch.from_numpy(poses_frame).float().to(device)
    trans_t = torch.from_numpy(trans_frame).float().to(device)

    # Get joints
    with torch.no_grad():
        vertices, joints = smpl_model.forward(betas=betas_t, poses=poses_t, trans=trans_t)

    return joints.cpu().numpy()[0]


def convert_addb_to_smpl_coords(joints):
    """Current conversion formula used in optimization"""
    return np.stack([joints[:, 0], joints[:, 2], -joints[:, 1]], axis=-1)


def test_alternative_conversions(addb_joints):
    """Test different coordinate conversion formulas"""
    conversions = {
        'Original (X, Z, -Y)': np.stack([addb_joints[:, 0], addb_joints[:, 2], -addb_joints[:, 1]], axis=-1),
        'No conversion (X, Y, Z)': addb_joints.copy(),
        'Simple swap (X, -Z, Y)': np.stack([addb_joints[:, 0], -addb_joints[:, 2], addb_joints[:, 1]], axis=-1),
        'Negated Y (X, -Y, Z)': np.stack([addb_joints[:, 0], -addb_joints[:, 1], addb_joints[:, 2]], axis=-1),
    }
    return conversions


def analyze_orientation(joints, name):
    """Analyze the orientation of joints to determine if standing upright"""
    pelvis = joints[0]  # Assume first joint is pelvis

    # Find highest and lowest joints
    y_values = joints[:, 1]
    highest_idx = np.argmax(y_values)
    lowest_idx = np.argmin(y_values)

    print(f"\n{name}:")
    print(f"  Pelvis position: X={pelvis[0]:.3f}, Y={pelvis[1]:.3f}, Z={pelvis[2]:.3f}")
    print(f"  Y-axis range: [{y_values.min():.3f}, {y_values.max():.3f}]")
    print(f"  Y-axis span: {y_values.max() - y_values.min():.3f}")
    print(f"  Highest joint (idx {highest_idx}): Y={joints[highest_idx, 1]:.3f}")
    print(f"  Lowest joint (idx {lowest_idx}): Y={joints[lowest_idx, 1]:.3f}")

    # Check if Y-axis is dominant (indicating upright standing)
    x_span = joints[:, 0].max() - joints[:, 0].min()
    y_span = joints[:, 1].max() - joints[:, 1].min()
    z_span = joints[:, 2].max() - joints[:, 2].min()

    print(f"  X span: {x_span:.3f}, Y span: {y_span:.3f}, Z span: {z_span:.3f}")

    dominant_axis = max([('X', x_span), ('Y', y_span), ('Z', z_span)], key=lambda x: x[1])
    print(f"  Dominant axis: {dominant_axis[0]} (span={dominant_axis[1]:.3f})")

    if dominant_axis[0] == 'Y' and y_span > 1.0:
        print(f"  ✓ Appears to be standing upright (Y-axis dominant)")
        return True
    else:
        print(f"  ✗ Does NOT appear to be standing upright")
        return False


def main():
    # Paths
    b3d_path = '/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Carter2023_Formatted_With_Arm/P020_split2/P020_split2.b3d'
    smpl_params_path = '/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames/Carter2023_Formatted_With_Arm_P020_split2/with_arm_carter2023_p020/smpl_params.npz'
    smpl_model_path = '/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl'

    frame_idx = 0

    print("="*80)
    print("COORDINATE SYSTEM VERIFICATION")
    print("="*80)

    # Read AddBiomechanics joints
    print("\n[1] Reading AddBiomechanics joints...")
    addb_joints = read_addb_joints(b3d_path, trial_idx=0, frame_idx=frame_idx)
    print(f"  AddB joints shape: {addb_joints.shape}")

    # Read SMPL joints
    print("\n[2] Reading SMPL optimized joints...")
    smpl_joints = read_smpl_joints(smpl_params_path, smpl_model_path, frame_idx)
    print(f"  SMPL joints shape: {smpl_joints.shape}")

    # Analyze original AddB orientation
    print("\n" + "="*80)
    print("ORIGINAL ADDBIOMECHANICS JOINTS (Z-up coordinate system)")
    print("="*80)
    addb_upright = analyze_orientation(addb_joints, "AddBiomechanics Original")

    # Analyze SMPL orientation
    print("\n" + "="*80)
    print("SMPL OPTIMIZED JOINTS (Y-up coordinate system)")
    print("="*80)
    smpl_upright = analyze_orientation(smpl_joints, "SMPL Optimized")

    # Test alternative conversions
    print("\n" + "="*80)
    print("TESTING ALTERNATIVE COORDINATE CONVERSIONS")
    print("="*80)

    conversions = test_alternative_conversions(addb_joints)

    for conv_name, converted_joints in conversions.items():
        is_upright = analyze_orientation(converted_joints, f"AddB with {conv_name}")

        if is_upright and conv_name != 'Original (X, Z, -Y)':
            print(f"\n  ⚠️  ALERT: {conv_name} produces upright orientation!")
            print(f"           This might be a better conversion than current one!")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if addb_upright and not smpl_upright:
        print("❌ PROBLEM CONFIRMED:")
        print("   - AddBiomechanics data shows upright standing pose")
        print("   - SMPL optimized results do NOT show upright pose")
        print("   - This indicates coordinate conversion is INCORRECT")
    elif addb_upright and smpl_upright:
        print("✓ Both show upright poses - coordinate system appears correct")
    else:
        print("⚠️  Unexpected result - need manual inspection")


if __name__ == '__main__':
    main()
