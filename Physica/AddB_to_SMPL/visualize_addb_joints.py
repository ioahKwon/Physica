#!/usr/bin/env python3
"""
AddBiomechanics 원본 joint positions 시각화
"""

import sys
import cv2
import numpy as np
from pathlib import Path

try:
    import nimblephysics as nimble
except ImportError:
    print("Error: nimblephysics is required")
    sys.exit(1)


def read_b3d_file(b3d_path, trial_idx=0):
    """
    Read AddBiomechanics .b3d file using nimblephysics

    Args:
        b3d_path: Path to .b3d file
        trial_idx: Trial index to read

    Returns:
        dict with keys: 'num_frames', 'num_joints', 'joint_names', 'joint_positions'
    """
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    # Get trial info
    num_trials = subject.getNumTrials()
    trial_length = subject.getTrialLength(trial_idx)
    trial_name = subject.getTrialName(trial_idx)

    print(f"  Num trials: {num_trials}")
    print(f"  Reading trial {trial_idx}: {trial_name}")
    print(f"  Trial length: {trial_length} frames")

    # Read skeleton
    skel = subject.readSkel(0)  # processingPass=0

    joint_names = []
    for i in range(skel.getNumJoints()):
        joint_names.append(skel.getJoint(i).getName())

    # Read all frames from this trial
    frames = subject.readFrames(
        trial=trial_idx,
        startFrame=0,
        numFramesToRead=trial_length,
        stride=1,
        includeSensorData=False,
        includeProcessingPasses=True
    )

    # Extract joint centers directly from processing passes
    joint_positions = []
    for frame in frames:
        if len(frame.processingPasses) > 0:
            # Joint centers are already computed in world coordinates
            jc = np.asarray(frame.processingPasses[0].jointCenters, dtype=np.float32)
            joint_positions.append(jc.reshape(-1, 3))

    joint_positions = np.array(joint_positions)  # (T, num_joints, 3)

    return {
        'num_frames': len(joint_positions),
        'num_joints': len(joint_names),
        'joint_names': joint_names,
        'joint_positions': joint_positions
    }


def visualize_addb_joints_2d(b3d_path, output_jpg, frame_idx=0):
    """
    AddBiomechanics joint positions를 2D로 시각화

    Args:
        b3d_path: .b3d 파일 경로
        output_jpg: 출력 이미지 경로
        frame_idx: 프레임 번호
    """
    print(f"Loading AddBiomechanics file: {b3d_path}")

    # Read B3D file
    data = read_b3d_file(b3d_path)

    print(f"  Total frames: {data['num_frames']}")
    print(f"  Number of joints: {data['num_joints']}")
    print(f"  Joint names: {data['joint_names']}")

    # Get joint positions for the specified frame
    joint_positions = data['joint_positions'][frame_idx]  # Shape: (num_joints, 3)
    joint_names = data['joint_names']

    print(f"\nFrame {frame_idx} joint positions:")
    for i, (name, pos) in enumerate(zip(joint_names, joint_positions)):
        print(f"  {i:2d} {name:20s}: X={pos[0]:7.3f}, Y={pos[1]:7.3f}, Z={pos[2]:7.3f}")

    # 2D projection (X-Y plane, Y-up)
    img_width, img_height = 1920, 1080
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Project to 2D (X, Y coordinates)
    joints_2d = joint_positions[:, [0, 1]].copy()
    joints_2d[:, 1] = -joints_2d[:, 1]  # Y축 반전 (이미지 좌표계)

    # Normalize
    range_val = max(
        joints_2d[:, 0].max() - joints_2d[:, 0].min(),
        joints_2d[:, 1].max() - joints_2d[:, 1].min()
    )

    scale = min(img_width, img_height) * 0.8 / range_val
    joints_2d_center = joints_2d.mean(axis=0)
    joints_2d = (joints_2d - joints_2d_center) * scale
    joints_2d[:, 0] += img_width // 2
    joints_2d[:, 1] += img_height // 2

    joints_2d = joints_2d.astype(int)

    # Draw kinematic tree connections (based on typical human skeleton structure)
    # These are approximate connections for AddBiomechanics joints
    connections = []

    # Try to infer connections based on joint names
    for i, name in enumerate(joint_names):
        if 'pelvis' in name.lower():
            pelvis_idx = i
        elif 'hip_l' in name.lower():
            l_hip_idx = i
            if pelvis_idx is not None:
                connections.append((pelvis_idx, l_hip_idx))
        elif 'hip_r' in name.lower():
            r_hip_idx = i
            if pelvis_idx is not None:
                connections.append((pelvis_idx, r_hip_idx))
        elif 'knee_l' in name.lower():
            l_knee_idx = i
            if 'l_hip_idx' in locals():
                connections.append((l_hip_idx, l_knee_idx))
        elif 'knee_r' in name.lower():
            r_knee_idx = i
            if 'r_hip_idx' in locals():
                connections.append((r_hip_idx, r_knee_idx))
        elif 'ankle_l' in name.lower():
            l_ankle_idx = i
            if 'l_knee_idx' in locals():
                connections.append((l_knee_idx, l_ankle_idx))
        elif 'ankle_r' in name.lower():
            r_ankle_idx = i
            if 'r_knee_idx' in locals():
                connections.append((r_knee_idx, r_ankle_idx))
        elif 'back' in name.lower():
            back_idx = i
            if pelvis_idx is not None:
                connections.append((pelvis_idx, back_idx))

    # Draw connections
    for i, j in connections:
        if i < len(joints_2d) and j < len(joints_2d):
            pt1 = tuple(joints_2d[i])
            pt2 = tuple(joints_2d[j])
            cv2.line(img, pt1, pt2, (100, 100, 255), 3)

    # Draw joints
    for i, (joint_2d, name) in enumerate(zip(joints_2d, joint_names)):
        cv2.circle(img, tuple(joint_2d), 8, (255, 0, 0), -1)
        # Joint label
        cv2.putText(img, f"{i}:{name[:10]}", (joint_2d[0] + 10, joint_2d[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Title
    cv2.putText(img, f"AddBiomechanics Joints - Frame {frame_idx} (Front View)",
               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

    # Save
    cv2.imwrite(output_jpg, img)
    print(f"\n✓ Saved: {output_jpg}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='AddBiomechanics joint 시각화')
    parser.add_argument('--b3d', type=str, required=True, help='.b3d 파일 경로')
    parser.add_argument('--output', type=str, required=True, help='출력 이미지 경로')
    parser.add_argument('--frame', type=int, default=0, help='프레임 번호')
    args = parser.parse_args()

    visualize_addb_joints_2d(args.b3d, args.output, args.frame)


if __name__ == '__main__':
    main()
