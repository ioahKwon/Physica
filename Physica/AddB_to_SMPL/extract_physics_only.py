#!/usr/bin/env python3
"""
Extract physics data from AddBiomechanics for already-processed subjects.
This script reads existing meta.json files, extracts the frame indices that were used,
and saves the corresponding physics data to physics_data.npz.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
try:
    import nimblephysics as nimble
except ImportError as exc:
    raise ImportError(
        "nimblephysics is required. Install with: pip install nimblephysics"
    ) from exc

def extract_physics_data(frames, processing_pass: int = 0) -> Dict:
    """Extract all available biomechanical/physics data from AddBiomechanics frames"""
    physics = {
        'ground_reaction_forces': [],
        'ground_reaction_moments': [],
        'center_of_pressure': [],
        'joint_torques': [],
        'joint_accelerations': [],
        'joint_velocities': [],
        'residual_forces': [],
        'residual_moments': [],
        'com_positions': [],
        'com_velocities': [],
        'com_accelerations': [],
    }

    for frame in frames:
        # Use processing pass data if available
        if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > 0:
            idx = min(processing_pass, len(frame.processingPasses) - 1)
            pp = frame.processingPasses[idx]

            # Ground reaction forces (GRF)
            if hasattr(pp, 'groundContactForce'):
                physics['ground_reaction_forces'].append(np.asarray(pp.groundContactForce, dtype=np.float32))
            elif hasattr(frame, 'grfBodyForce'):
                physics['ground_reaction_forces'].append(np.asarray(frame.grfBodyForce, dtype=np.float32))
            else:
                physics['ground_reaction_forces'].append(None)

            # Ground reaction moments
            if hasattr(pp, 'groundContactTorque'):
                physics['ground_reaction_moments'].append(np.asarray(pp.groundContactTorque, dtype=np.float32))
            elif hasattr(frame, 'grfBodyTorque'):
                physics['ground_reaction_moments'].append(np.asarray(frame.grfBodyTorque, dtype=np.float32))
            else:
                physics['ground_reaction_moments'].append(None)

            # Center of pressure
            if hasattr(pp, 'groundContactCenterOfPressure'):
                physics['center_of_pressure'].append(np.asarray(pp.groundContactCenterOfPressure, dtype=np.float32))
            elif hasattr(frame, 'cop'):
                physics['center_of_pressure'].append(np.asarray(frame.cop, dtype=np.float32))
            else:
                physics['center_of_pressure'].append(None)

            # Joint torques (tau)
            if hasattr(pp, 'tau'):
                physics['joint_torques'].append(np.asarray(pp.tau, dtype=np.float32))
            else:
                physics['joint_torques'].append(None)

            # Joint accelerations (ddq)
            if hasattr(pp, 'acc'):
                physics['joint_accelerations'].append(np.asarray(pp.acc, dtype=np.float32))
            elif hasattr(pp, 'ddq'):
                physics['joint_accelerations'].append(np.asarray(pp.ddq, dtype=np.float32))
            else:
                physics['joint_accelerations'].append(None)

            # Joint velocities (dq)
            if hasattr(pp, 'vel'):
                physics['joint_velocities'].append(np.asarray(pp.vel, dtype=np.float32))
            elif hasattr(pp, 'dq'):
                physics['joint_velocities'].append(np.asarray(pp.dq, dtype=np.float32))
            else:
                physics['joint_velocities'].append(None)

            # Residual forces
            if hasattr(pp, 'residualWrenchInRootFrame'):
                physics['residual_forces'].append(np.asarray(pp.residualWrenchInRootFrame[:3], dtype=np.float32))
            else:
                physics['residual_forces'].append(None)

            # Residual moments
            if hasattr(pp, 'residualWrenchInRootFrame'):
                physics['residual_moments'].append(np.asarray(pp.residualWrenchInRootFrame[3:], dtype=np.float32))
            else:
                physics['residual_moments'].append(None)

            # Center of mass
            if hasattr(pp, 'comPos'):
                physics['com_positions'].append(np.asarray(pp.comPos, dtype=np.float32))
            else:
                physics['com_positions'].append(None)

            if hasattr(pp, 'comVel'):
                physics['com_velocities'].append(np.asarray(pp.comVel, dtype=np.float32))
            else:
                physics['com_velocities'].append(None)

            if hasattr(pp, 'comAcc'):
                physics['com_accelerations'].append(np.asarray(pp.comAcc, dtype=np.float32))
            else:
                physics['com_accelerations'].append(None)
        else:
            # No processing passes - fill with None
            for key in physics.keys():
                physics[key].append(None)

    # Convert lists to numpy arrays (or None if all are None)
    result = {}
    for key, values in physics.items():
        if all(v is None for v in values):
            result[key] = None
        else:
            # Filter out None values and stack
            valid_values = [v if v is not None else np.zeros_like(next(vv for vv in values if vv is not None))
                           for v in values]
            result[key] = np.array(valid_values, dtype=np.float32)

    return result


def process_subject(subject_dir: Path, b3d_path: str, trial: int = 0, processing_pass: int = 0):
    """Extract and save physics data for a single subject"""

    # Check if physics data already exists
    physics_file = subject_dir / "physics_data.npz"
    if physics_file.exists():
        print(f"  SKIP (physics data exists)")
        return True

    # Read meta.json to get frame indices
    meta_file = subject_dir / "meta.json"
    if not meta_file.exists():
        print(f"  ERROR: meta.json not found")
        return False

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    # Get frame sampling info
    frame_sampling = meta.get('frame_sampling', {})
    if frame_sampling.get('enabled', False):
        # Subject was processed with frame sampling
        selected_indices = np.array(frame_sampling['selected_frame_indices'])
        original_frames = frame_sampling['original_num_frames']
    else:
        # No frame sampling - use all frames
        num_frames = meta['num_frames']
        selected_indices = np.arange(num_frames)
        original_frames = num_frames

    # Load AddBiomechanics data
    try:
        subj = nimble.biomechanics.SubjectOnDisk(b3d_path)
        total = subj.getTrialLength(trial)

        # Read ALL frames with physics data
        frames = subj.readFrames(
            trial=trial,
            startFrame=0,
            numFramesToRead=total,
            includeProcessingPasses=True,
            includeSensorData=True,  # Enable physics data
            stride=1,
            contactThreshold=1.0
        )

        if len(frames) == 0:
            print(f"  ERROR: Failed to read frames from .b3d")
            return False

        # Extract physics data for ALL frames
        physics_data_full = extract_physics_data(frames, processing_pass)

        # Sample physics data using SAME indices as SMPL optimization
        physics_data = {}
        for key, data in physics_data_full.items():
            if data is not None and len(data) > 0:
                physics_data[key] = data[selected_indices]
            else:
                physics_data[key] = None

        # Save physics data
        physics_save_dict = {k: v for k, v in physics_data.items() if v is not None}
        if physics_save_dict:
            np.savez(physics_file, **physics_save_dict)
            print(f"  ✓ Saved {len(physics_save_dict)} physics fields")
            return True
        else:
            print(f"  WARNING: No physics data available")
            return False

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing processed subjects')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Root directory of AddBiomechanics dataset')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)

    # Find all processed subjects (meta.json is in subdirectory like with_arm_xxx/)
    subject_dirs = []
    for parent_dir in output_dir.iterdir():
        if parent_dir.is_dir():
            # Look for meta.json in subdirectories
            for sub_dir in parent_dir.iterdir():
                if sub_dir.is_dir() and (sub_dir / "meta.json").exists():
                    subject_dirs.append(sub_dir)
                    break

    print(f"{'='*80}")
    print(f"EXTRACT PHYSICS DATA FOR COMPLETED SUBJECTS")
    print(f"{'='*80}")
    print(f"Output dir: {output_dir}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Found {len(subject_dirs)} processed subjects\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, subject_dir in enumerate(subject_dirs, 1):
        # Parse subject info from meta.json
        with open(subject_dir / "meta.json", 'r') as f:
            meta = json.load(f)

        b3d_path = meta['b3d']
        subject_name = subject_dir.name

        print(f"[{i}/{len(subject_dirs)}] {subject_name}")

        # Check if physics data already exists
        if (subject_dir / "physics_data.npz").exists():
            print(f"  SKIP (already exists)")
            skip_count += 1
            continue

        # Process subject
        result = process_subject(subject_dir, b3d_path)
        if result:
            success_count += 1
        else:
            error_count += 1

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total subjects: {len(subject_dirs)}")
    print(f"Successful: {success_count}")
    print(f"Skipped (exists): {skip_count}")
    print(f"Errors: {error_count}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
